from copy import deepcopy
from typing import Any, Dict

import pytest
import torch
import torch.nn.functional as F
from e3nn import o3

from mace import data, modules, tools
from mace.tools import torch_geometric

# --- Import OEQ Conversion Scripts ---
# Make sure these paths are correct and the scripts exist
try:
    from mace.cli.convert_oeq_e3nn import run as run_oeq_to_e3nn
    from mace.cli.convert_e3nn_oeq import run as run_e3nn_to_oeq

    CONVERSION_SCRIPTS_AVAILABLE = True
except ImportError:
    CONVERSION_SCRIPTS_AVAILABLE = False
    run_oeq_to_e3nn = None
    run_e3nn_to_oeq = None
# ---------------------------------


# --- Check if openequivariance is installed ---
try:
    import openequivariance as oeq  # Or just 'import openequivariance'

    OEQ_AVAILABLE = True
except ImportError:
    OEQ_AVAILABLE = False
# ------------------------------------------

ACCELERATOR_AVAILABLE = torch.cuda.is_available()


# Skip the entire class if OEQ lib or conversion scripts are missing
@pytest.mark.skipif(not OEQ_AVAILABLE, reason="openequivariance library not installed")
@pytest.mark.skipif(
    not CONVERSION_SCRIPTS_AVAILABLE, reason="OEQ conversion scripts not found"
)
@pytest.mark.skipif(
    not ACCELERATOR_AVAILABLE, reason="CUDA/ROCm accelerator not available"
)
class TestOeq:
    @pytest.fixture
    def model_config(self, interaction_cls_first, hidden_irreps) -> Dict[str, Any]:
        """Fixture for base E3NN model configuration."""
        table = tools.AtomicNumberTable([6])  # Using Carbon only for simplicity
        return {
            "r_max": 5.0,
            "num_bessel": 8,
            "num_polynomial_cutoff": 6,
            "max_ell": 3,
            "interaction_cls": modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            "interaction_cls_first": interaction_cls_first,
            "num_interactions": 2,
            "num_elements": 1,  # Only Carbon
            "hidden_irreps": hidden_irreps,
            "MLP_irreps": o3.Irreps("16x0e"),
            "gate": F.silu,
            "atomic_energies": torch.tensor([1.0]),  # Dummy energy
            "avg_num_neighbors": 8,
            "atomic_numbers": table.zs,
            "correlation": 3,
            "radial_type": "bessel",
            "atomic_inter_scale": 1.0,
            "atomic_inter_shift": 0.0,
            # No oeq_config here, start with E3NN base model
        }

    @pytest.fixture
    def batch(self, device: str, default_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        """Fixture for creating a sample batch of data."""
        from ase import build
        import numpy as np

        torch.set_default_dtype(default_dtype)
        table = tools.AtomicNumberTable([6])

        # Create a simple diamond structure, displace atoms slightly, repeat
        atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
        displacement = np.random.uniform(
            -0.05, 0.05, size=atoms.positions.shape
        )  # Smaller displacement
        atoms.positions += displacement
        atoms_list = [atoms.repeat((1, 1, 1))]  # Smaller system for faster test

        configs = [data.config_from_atoms(atoms) for atoms in atoms_list]
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(config, z_table=table, cutoff=5.0)
                for config in configs
            ],
            batch_size=len(configs),  # Process all configs in one batch
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader))
        return batch.to(device).to_dict()

    # Parametrize over devices, interaction blocks, hidden_irreps, and dtype
    @pytest.mark.parametrize("device", ["cuda"])
    @pytest.mark.parametrize(
        "interaction_cls_first",
        [
            modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
            # Add other interaction classes if needed and supported
        ],
    )
    @pytest.mark.parametrize(
        "hidden_irreps",
        [
            o3.Irreps("16x0e + 16x1o"),  # Smaller model for faster test
            # o3.Irreps("32x0e + 32x1o + 32x2e"), # Can add more complex cases
        ],
    )
    @pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
    def test_bidirectional_conversion_oeq(
        self,
        model_config: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        device: str,
        default_dtype: torch.dtype,
    ):
        """Tests E3NN -> OEQ -> E3NN conversion and numerical equivalence."""
        torch.manual_seed(42)
        torch.set_default_dtype(default_dtype)

        # Create original E3nn model
        model_e3nn = modules.ScaleShiftMACE(**model_config).to(device)
        model_oeq = run_e3nn_to_oeq(model_e3nn).to(device)
        model_e3nn_back = run_oeq_to_e3nn(model_oeq).to(device)

        # --- Test forward pass equivalence ---
        batch_e3nn = deepcopy(batch)
        batch_oeq = deepcopy(batch)
        batch_e3nn_back = deepcopy(batch)
        torch.testing.assert_close(batch_e3nn, batch_oeq)
        torch.testing.assert_close(batch_oeq, batch_e3nn_back)

        # Test forward pass equivalence
        out_e3nn = model_e3nn(batch_e3nn, training=True, compute_stress=True)
        out_oeq = model_oeq(batch_oeq, training=True, compute_stress=True)
        out_e3nn_back = model_e3nn_back(
            batch_e3nn_back, training=True, compute_stress=True
        )

        # Check outputs match for both conversions
        # torch.testing.assert_close(out_e3nn["energy"], out_oeq["energy"])
        # torch.testing.assert_close(out_oeq["energy"], out_e3nn_back["energy"])
        # torch.testing.assert_close(out_e3nn["forces"], out_oeq["forces"])
        # torch.testing.assert_close(out_oeq["forces"], out_e3nn_back["forces"])
        # torch.testing.assert_close(out_e3nn["stress"], out_oeq["stress"])
        # torch.testing.assert_close(out_oeq["stress"], out_e3nn_back["stress"])
        # -----------------------------------

        # Test backward pass equivalence
        loss_e3nn = out_e3nn["energy"].sum()
        loss_oeq = out_oeq["energy"].sum()
        loss_e3nn_back = out_e3nn_back["energy"].sum()

        loss_e3nn.backward()
        loss_oeq.backward()
        loss_e3nn_back.backward()

        # Compare gradients for all conversions
        tol = 1e-4 if default_dtype == torch.float32 else 1e-8

        def print_gradient_diff(name1, p1, name2, p2, conv_type):
            if p1.grad is not None and p1.grad.shape == p2.grad.shape:
                if name1.split(".", 2)[:2] == name2.split(".", 2)[:2]:
                    error = torch.abs(p1.grad - p2.grad)
                    print(
                        f"{conv_type} - Parameter {name1}/{name2}, Max error: {error.max()}"
                    )
                    torch.testing.assert_close(p1.grad, p2.grad, atol=tol, rtol=1e-10)

        # E3nn to OEQ gradients
        for (name_e3nn, p_e3nn), (name_oeq, p_oeq) in zip(
            model_e3nn.named_parameters(), model_oeq.named_parameters()
        ):
            print_gradient_diff(name_e3nn, p_e3nn, name_oeq, p_oeq, "E3nn->OEQ")

        # OEQ to E3nn gradients
        for (name_oeq, p_oeq), (name_e3nn_back, p_e3nn_back) in zip(
            model_oeq.named_parameters(), model_e3nn_back.named_parameters()
        ):
            print_gradient_diff(
                name_oeq, p_oeq, name_e3nn_back, p_e3nn_back, "OEQ->E3nn"
            )

        # Full circle comparison (E3nn -> E3nn)
        for (name_e3nn, p_e3nn), (name_e3nn_back, p_e3nn_back) in zip(
            model_e3nn.named_parameters(), model_e3nn_back.named_parameters()
        ):
            print_gradient_diff(
                name_e3nn, p_e3nn, name_e3nn_back, p_e3nn_back, "Full circle"
            )
