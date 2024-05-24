from omegaconf import DictConfig
from hydra.utils import instantiate
from src.modules.TemporalConditionalSplineFlow import TemporalConditionalSplineFlow

def generate_model(cfg: DictConfig):
    
    lag = cfg.lag
    num_nodes = cfg.num_nodes
    data_dim = cfg.data_dim
    num_workers = cfg.num_workers
    aggregated_graph = cfg.aggregated_graph

    if cfg.model == 'pcmci' or cfg.model == 'varlingam' \
        or cfg.model == 'dynotears':

        trainer = instantiate(cfg.trainer, 
                                num_workers=num_workers,
                                lag=lag, 
                                num_nodes=num_nodes,
                                data_dim=data_dim,
                                aggregated_graph=aggregated_graph)
        
    else:
        
        multi_graph = cfg.model == 'mcd'
        if multi_graph:
            num_graphs = cfg.trainer.num_graphs

        if 'decoder' in cfg:
            # generate the decoder
            if not multi_graph:    
                causal_decoder = instantiate(cfg.decoder, 
                                        lag=lag, 
                                        num_nodes=num_nodes,
                                        data_dim=data_dim)
            else:
                causal_decoder = instantiate(cfg.decoder, 
                                        lag=lag, 
                                        num_nodes=num_nodes,
                                        data_dim=data_dim,
                                        num_graphs=num_graphs)
                
        if 'likelihood_loss' in cfg.trainer and cfg.trainer.likelihood_loss == 'flow':
            # create hypernet
            if not multi_graph:
                hypernet = instantiate(cfg.hypernet, 
                                lag=lag, 
                                num_nodes=num_nodes,
                                data_dim=data_dim)
            else:
                hypernet = instantiate(cfg.hypernet, 
                                lag=lag, 
                                num_nodes=num_nodes,
                                data_dim=data_dim,
                                num_graphs=num_graphs)
                
            tcsf = TemporalConditionalSplineFlow(hypernet=hypernet)
        else:
            hypernet = None
            tcsf = None

        # create auglag config
        if cfg.trainer.training_procedure == 'auglag':
            training_config = instantiate(cfg.auglag_config)
        
        if cfg.model == 'rhino':
            trainer = instantiate(cfg.trainer, 
                                num_workers=num_workers,
                                lag=lag, 
                                num_nodes=num_nodes,
                                data_dim=data_dim,
                                causal_decoder=causal_decoder,
                                tcsf=tcsf,
                                training_config=training_config,
                                aggregated_graph=aggregated_graph)
        elif cfg.model == 'mcd':
            trainer = instantiate(cfg.trainer, 
                                num_workers=num_workers,
                                lag=lag, 
                                num_nodes=num_nodes,
                                data_dim=data_dim,
                                num_graphs=num_graphs,
                                causal_decoder=causal_decoder,
                                tcsf=tcsf,
                                training_config=training_config,
                                aggregated_graph=aggregated_graph)
        
    return trainer
