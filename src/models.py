import torch
import torch.nn as nn
from sklearn.cluster import KMeans

def _create_encoder_block(full_dims, dropout):
    """Helper function to create an encoder block."""
    layers = []
    for i, (in_dim, out_dim) in enumerate(zip(full_dims[:-1], full_dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        # Apply BatchNorm, ReLU, Dropout to all but the last layer
        if i < len(full_dims) - 1:
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

class MIC(nn.Module):
    """
    Multi-omics Integration Clustering (MIC) Model.
    Dynamically constructs the network based on provided dimensions.
    """
    def __init__(self, input_dims, encoder_dims, integration_dims, latent_dim, decoder_dims, clinical_output_dim, cluster_num=4, dropout=0.2):
        super(MIC, self).__init__()
        
        self.cluster_num = cluster_num

        # Create encoders for each omics type
        self.genotype_encoder = _create_encoder_block([input_dims['genotype']] + encoder_dims, dropout)
        self.proteome_encoder = _create_encoder_block([input_dims['proteome']] + encoder_dims, dropout)
        self.metabolite_encoder = _create_encoder_block([input_dims['metabolite']] + encoder_dims, dropout)

        # Create the integration block (REVISED LOGIC TO BE 100% IDENTICAL TO ORIGINAL)
        integrator_input_dim = encoder_dims[-1] * 3
        
        integrator_layers = []
        # This loop structure perfectly replicates the original logic.
        # It applies activation/dropout only to the defined hidden layers.
        current_dim = integrator_input_dim
        for hidden_dim in integration_dims:
            integrator_layers.append(nn.Linear(current_dim, hidden_dim))
            integrator_layers.append(nn.BatchNorm1d(hidden_dim))
            integrator_layers.append(nn.ReLU())
            integrator_layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # Manually add the final linear layer to produce the latent space.
        # No activation/dropout is applied after this layer.
        integrator_layers.append(nn.Linear(current_dim, latent_dim))

        self.latent_integrator = nn.Sequential(*integrator_layers)
        
        # Create the clinical decoder
        decoder_full_dims = [latent_dim] + decoder_dims + [clinical_output_dim]
        
        decoder_layers = []
        for i, (in_dim, out_dim) in enumerate(zip(decoder_full_dims[:-1], decoder_full_dims[1:])):
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < len(decoder_full_dims) - 2:
                decoder_layers.append(nn.ReLU())
        self.clinical_decoder = nn.Sequential(*decoder_layers)

        # Initialize centroids and weights
        self.centroid = nn.Parameter(torch.zeros(self.cluster_num, latent_dim))
        nn.init.normal_(self.centroid, mean=0, std=0.01)
        self.apply(self._init_weight)

    def forward(self, genotype, proteome, metabolite):
        z = self.encode(genotype, proteome, metabolite)
        return self.clinical_decoder(z)

    def encode(self, genotype, proteome, metabolite):
        genotype_latent = self.genotype_encoder(genotype)
        proteome_latent = self.proteome_encoder(proteome)
        metabolite_latent = self.metabolite_encoder(metabolite)
        combined_latent = torch.cat((genotype_latent, proteome_latent, metabolite_latent), dim=1)
        return self.latent_integrator(combined_latent)

    def get_latent_space(self, data_loader, device="cpu"):
        self.eval()
        latent_space = []
        with torch.no_grad():
            for batch in data_loader:
                # Handle loaders with or without labels
                inputs = batch[:3]
                x1, x2, x3 = [d.to(device) for d in inputs]
                z = self.encode(x1, x2, x3)
                latent_space.append(z.cpu())
        return torch.cat(latent_space, dim=0)

    def k_means_clustering(self, data_loader, n_init=100, device="cpu"):
        latent_space = self.get_latent_space(data_loader, device=device).numpy()
        kmeans = KMeans(n_clusters=self.cluster_num, init="k-means++", n_init=n_init, random_state=214)
        labels = kmeans.fit_predict(latent_space)
        self.centroid.data = torch.tensor(kmeans.cluster_centers_).to(device)
        return labels
    
    def soft_cluster_assignment(self, z, alpha=1.0):
        diff = torch.sum((z.unsqueeze(1) - self.centroid) ** 2, dim=2)
        numerator = (1.0 + (diff / alpha)) ** (-(alpha + 1.0) / 2.0)
        q = numerator / torch.sum(numerator, dim=1, keepdim=True)
        return q
    
    def _init_weight(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight)
            nn.init.constant_(layer.bias, 0)