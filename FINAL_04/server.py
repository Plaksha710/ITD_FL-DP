import flwr as fl
from typing import Dict, Optional, List, Tuple
from flwr.common import Parameters, Scalar, FitRes, EvaluateRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.server.strategy.strategy import ClientProxy 
from flwr.common.logger import log
from logging import INFO
import numpy as np
import pandas as pd 

# ----------------------------------------------------
# 1. Non-DP Strategy: LoggingFedAvg (Scenario 3)
# ----------------------------------------------------
class LoggingFedAvg(FedAvg):
    """
    Standard FedAvg strategy with weighted metric logging. 
    It explicitly saves global_weights and aggregates evaluation metrics manually (for consistency).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []
        self.global_weights = None 

    def aggregate_fit(self, rnd, results, failures):
        # Call the parent class's aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        # Store the aggregated weights for saving the model later
        self.global_weights = aggregated_parameters
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException | Tuple[ClientProxy, BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        
        if not results:
            return None, {}

        # 1. Calculate weighted loss
        weighted_losses = []
        examples = []
        for _, res in results: 
            weighted_losses.append(res.loss * res.num_examples)
            examples.append(res.num_examples)
        
        total_samples = sum(examples)
        weighted_loss = sum(weighted_losses) / total_samples
        
        # 2. Calculate WEIGHTED AVERAGE for ALL other metrics
        aggregated_metrics = {}
        first_res = results[0][1] 
        
        if first_res.metrics:
            metric_keys = first_res.metrics.keys()
            
            for key in metric_keys:
                weighted_sum = sum(
                    res.metrics.get(key, 0.0) * res.num_examples
                    for _, res in results
                )
                aggregated_metrics[key] = weighted_sum / total_samples

        log(INFO, f"Round {rnd} weighted aggregated metrics: {aggregated_metrics}")
        
        self.history.append({
            "round": rnd,
            "loss": weighted_loss,
            **aggregated_metrics
        })

        return weighted_loss, aggregated_metrics

# ----------------------------------------------------
# 2. DP Strategy: DpLoggingFedAvg (Scenario 4)
# ----------------------------------------------------
class DpLoggingFedAvg(FedAvg):
    """
    Federated Averaging strategy with Differential Privacy (Noise and Clipping)
    and custom weighted metric logging.
    """
    
    def __init__(self, *args, clip_norm=None, noise_multiplier=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters: Optional[Parameters] = None 
        
        self.history = []
        self.global_weights = None
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        log(INFO, f"DP Config: Clip Norm (C)={self.clip_norm}, Noise Multiplier (sigma)={self.noise_multiplier}")

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        if not results:
            return None, {}
        
        current_parameters = self.parameters
        
        if current_parameters is None or rnd == 1:
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
            self.parameters = aggregated_parameters 
            self.global_weights = aggregated_parameters
            return aggregated_parameters, aggregated_metrics
        
        current_weights = parameters_to_ndarrays(current_parameters)
        updates = []
        
        # 1. Calculate and process updates from clients (including Clipping)
        for client_proxy, fit_res in results:
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            
            # Calculate the update: Update = W_new - W_old
            update = [w_new - w_old for w_new, w_old in zip(client_weights, current_weights)]
            
            # --- DP: L2-Clipping (on the update norm) ---
            if self.clip_norm is not None and self.clip_norm > 0:
                l2_norm = np.sqrt(sum([np.sum(u**2) for u in update]))
                if l2_norm > self.clip_norm:
                    scale = self.clip_norm / l2_norm
                    update = [u * scale for u in update]
            
            updates.append((update, fit_res.num_examples))
        
        # 2. Aggregate the clipped updates (Weighted Average)
        weights = [num_examples for _, num_examples in updates]
        total_weight = sum(weights)
        
        aggregated_update = [
            np.sum([u * weight for u, weight in zip(layer_updates, weights)], axis=0) / total_weight
            for layer_updates in zip(*[update for update, _ in updates])
        ]
        
        # 3. --- DP: Add Gaussian Noise (on the aggregated update) ---
        if self.noise_multiplier is not None and self.noise_multiplier > 0:
            sensitivity = self.clip_norm if self.clip_norm is not None else 1.0 
            scale = self.noise_multiplier * sensitivity / len(results)
            
            for i in range(len(aggregated_update)):
                shape = aggregated_update[i].shape
                noise = np.random.normal(0, scale, size=shape)
                aggregated_update[i] += noise

        # 4. Apply the aggregated update to the global model
        new_weights = [w_old + a_update for w_old, a_update in zip(current_weights, aggregated_update)]
        aggregated_parameters = ndarrays_to_parameters(new_weights)
        
        self.parameters = aggregated_parameters 
        self.global_weights = aggregated_parameters
        
        return aggregated_parameters, {}

    # aggregate_evaluate (Same logging logic as the Non-DP version)
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[ClientProxy, EvaluateRes]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        
        if not results:
            return None, {}

        # 1. Calculate weighted loss
        weighted_losses = []
        examples = []
        for _, res in results: 
            weighted_losses.append(res.loss * res.num_examples)
            examples.append(res.num_examples)
        
        total_samples = sum(examples)
        weighted_loss = sum(weighted_losses) / total_samples
        
        # 2. Calculate WEIGHTED AVERAGE for ALL other metrics
        aggregated_metrics = {}
        first_res = results[0][1] 
        
        if first_res.metrics:
            metric_keys = first_res.metrics.keys()
            
            for key in metric_keys:
                weighted_sum = sum(
                    res.metrics.get(key, 0.0) * res.num_examples
                    for _, res in results
                )
                aggregated_metrics[key] = weighted_sum / total_samples

        log(INFO, f"Round {rnd} weighted aggregated metrics: {aggregated_metrics}")
        
        self.history.append({
            "round": rnd,
            "loss": weighted_loss,
            **aggregated_metrics
        })

        return weighted_loss, aggregated_metrics
