import torch

class PermPredictor(torch.nn.Module):
    def __init__(self, prop_vec_size, dim_seq, dropout_pair):
        super().__init__()

        predictor_list = []
        predictor_list.append(torch.nn.Linear(prop_vec_size, dim_seq[0]))
        predictor_list.append(torch.nn.ReLU())
        predictor_list.append(torch.nn.Dropout(p = dropout_pair[0]))
        for i, dim_size in enumerate(dim_seq[1:]):
            predictor_list.append(torch.nn.Linear(dim_seq[i], dim_seq[i+1]))
            predictor_list.append(torch.nn.ReLU())
            predictor_list.append(torch.nn.Dropout(p = dropout_pair[1]))
        predictor_list.append(torch.nn.Linear(dim_seq[-1], 1))
        
        self.predictor = torch.nn.Sequential(*predictor_list)

    def forward(self, prop_vecs):
        predictions = self.predictor(prop_vecs).flatten()
        
        return predictions
