import torch


def compute_velocity_field(input: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute the velocity field along the a dimension of any input tensor

    Parameters:
        - tensor (torch.Tensor): Input tensor
        - dim (int): The axis along which the velocity field is computed

    Returns:
        torch.Tensor
    """
    output: torch.Tensor = (input ** 2).sum(dim=dim, keepdim=True) ** 0.5
    assert output.shape[dim] == 1
    return output


