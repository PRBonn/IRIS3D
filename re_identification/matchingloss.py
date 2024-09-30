import torch

class MatchingLoss(torch.nn.Module):
    def __init__(self, lambda_A=0.08):
        super(MatchingLoss, self).__init__()
        self.lambda_A = lambda_A

    def forward(self, predictions):

        predsoft = torch.nn.functional.softmax(predictions, dim=-1)

        # Assignment Constraint Loss
        # Sum of probabilities for each object at time t+1 should be at most 1
        row_sum = torch.sum(predsoft[:, 1:], dim=1)  # Exclude "no match" column for row sum
        row_constraint_loss = torch.sum(torch.abs(row_sum - 1))

        # Sum of probabilities for each object at time t should be at most 1
        col_sum = torch.sum(predsoft[:, 1:], dim=0)  # Exclude "no match" column for col sum
        col_constraint_loss = torch.sum(torch.abs(col_sum - 1))

        assignment_constraint_loss = self.lambda_A * (row_constraint_loss + col_constraint_loss)

        return assignment_constraint_loss
