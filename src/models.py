import torch
import torch.nn as nn

def _create_encoder_block(full_dims, dropout):
    """Helper function to create an encoder block."""
    layers = []
    for i, (in_dim, out_dim) in enumerate(zip(full_dims[:-1], full_dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        # Match the submitted model: normalize and activate every encoder layer.
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

        # Create the integration block.
        integrator_input_dim = encoder_dims[-1] * 3
        
        integrator_layers = []
        current_dim = integrator_input_dim
        for hidden_dim in integration_dims:
            integrator_layers.append(nn.Linear(current_dim, hidden_dim))
            integrator_layers.append(nn.BatchNorm1d(hidden_dim))
            integrator_layers.append(nn.ReLU())
            integrator_layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # No activation or dropout is applied to the latent representation.
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

    def soft_cluster_assignment(self, z, alpha=1.0):
        diff = torch.sum((z.unsqueeze(1) - self.centroid) ** 2, dim=2)
        numerator = (1.0 + (diff / alpha)) ** (-(alpha + 1.0) / 2.0)
        q = numerator / torch.sum(numerator, dim=1, keepdim=True)
        return q
    
    def _init_weight(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight)
            nn.init.constant_(layer.bias, 0)


def build_mic(input_dims, params, config_module):
    return MIC(
        input_dims=input_dims,
        encoder_dims=config_module.ENCODER_DIMS,
        integration_dims=config_module.INTEGRATION_DIMS,
        latent_dim=config_module.LATENT_DIM,
        decoder_dims=list(params["decoder_dim"]),
        clinical_output_dim=config_module.CLINICAL_OUTPUT_DIM,
        cluster_num=config_module.NUM_CLUSTERS,
        dropout=float(params["dropout"]),
    )


class MICSingle(nn.Module):
    def __init__(self, input_dim, params, latent_dim=16, clinical_dim=5,
                 cluster_num=4):
        super().__init__()
        dropout = float(params["dropout"])
        # Preserve the finalized single-omics architecture, including the
        # additional 16-to-16 encoder layer before the latent representation.
        encoder_dims = [input_dim, 128, 32, 16, latent_dim]
        encoder_layers = []
        for in_dim, out_dim in zip(encoder_dims[:-1], encoder_dims[1:]):
            encoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_dims = [latent_dim] + list(params["decoder_dim"]) + [clinical_dim]
        decoder_layers = []
        for index, (in_dim, out_dim) in enumerate(
            zip(decoder_dims[:-1], decoder_dims[1:])
        ):
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            if index < len(decoder_dims) - 2:
                decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)
        self.centroid = nn.Parameter(torch.randn(cluster_num, latent_dim) * 0.01)
        self.apply(self._init_weight)

    def encode(self, inputs):
        return self.encoder(inputs)

    def forward(self, inputs):
        return self.decoder(self.encode(inputs))

    @staticmethod
    def _init_weight(layer):
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
