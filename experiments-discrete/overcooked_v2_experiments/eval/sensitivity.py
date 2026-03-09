
import jax
import jax.numpy as jnp
import numpy as np
import csv
import os
import copy
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.ppo.models.e3t import DiClusterEncoder

def analyze_sensitivity_pairing(
    policy_pairing,
    layout_name,
    key,
    env_kwargs,
    save_path, 
    ego_direction=False,
    num_steps=50,
):
    """
    Runs an episode for num_steps, and at intervals performs a latent sweep sensitivity analysis.
    Uses policy.network.apply directly to inject latents, bypassing compute_action if necessary.
    """
    print(f"[Sensitivity] Starting analysis for {layout_name}...")
    
    # Setup Environment
    env = OvercookedV2(layout=layout_name, **env_kwargs)
    
    # Initialize policies
    rng = key
    rng, reset_key = jax.random.split(rng)
    
    # Assume 2 agents
    num_agents = 2
    
    # Reset
    obs, env_state = env.reset(reset_key)

    # History buffers for real-z computation
    obs_shape = env.observation_space().shape
    history_len = 5
    obs_history = {
        f"agent_{i}": jnp.zeros((history_len, *obs_shape)) for i in range(num_agents)
    }
    act_history = {
        f"agent_{i}": jnp.zeros((history_len,), dtype=jnp.int32) for i in range(num_agents)
    }

    di_cluster_encoder = DiClusterEncoder()
    ov_type = "ov2"
    
    # H-States
    hstates = [
        policy_pairing.policies[i].init_hstate(1, rng) 
        for i in range(num_agents)
    ]
    
    results = []
    
    # Loop
    for step in range(1, num_steps + 1):
        # 1. Select Actions (Simulation Step)
        actions = {}
        
        # Compute real latent from history for simulation step
        # Stack history for DiClusterEncoder
        obs_hist_batch = jnp.stack([obs_history[f"agent_{i}"] for i in range(num_agents)])
        act_hist_batch = jnp.stack([act_history[f"agent_{i}"] for i in range(num_agents)])
        z_ego_onehot, z_part_onehot = di_cluster_encoder.apply({}, obs_hist_batch, act_hist_batch, ov_type)

        latents_sim = []
        for i in range(num_agents):
            z_p = z_part_onehot[i]
            z_e = z_ego_onehot[i]
            if ego_direction:
                pp = jnp.concatenate([z_p, z_e], axis=-1)
            else:
                pp = z_p
            latents_sim.append(pp)
            
        next_hstates = []
        
        for i in range(num_agents):
            agent_id = env.agents[i]
            curr_obs = obs[agent_id]
            policy = policy_pairing.policies[i]
            
            # Prepare inputs for apply
            # Obs: (H, W, C) -> (1, 1, H, W, C) (Time, Batch, ...)
            obs_in = jnp.expand_dims(jnp.expand_dims(curr_obs, 0), 0)
            done_in = jnp.zeros((1, 1))
            ac_in = (obs_in, done_in)
            
            # Prepare latent input
            # (Dim) -> (1, 1, Dim)
            pp_in = latents_sim[i][jnp.newaxis, jnp.newaxis, ...]
            
            # Forward pass
            h_state = hstates[i]
            new_h, pi, _ = policy.network.apply(
                policy.params,
                h_state,
                ac_in,
                partner_prediction=pp_in
            )
            
            rng, action_key = jax.random.split(rng)
            if policy.stochastic:
                action_idx = pi.sample(seed=action_key)
            else:
                action_idx = jnp.argmax(pi.probs, axis=-1)
                
            action_idx = action_idx.squeeze() # (1, 1) -> scalar
            
            actions[agent_id] = action_idx
            next_hstates.append(new_h)
            
        # 2. Sensitivity Snapshot (Step 10, 20, 30, 40, 50)
        if step % 10 == 0:
            print(f"  [Analysis] Step {step}")
            
            # Analyze Agent 0 (Ego)
            agent_idx = 0
            ego_obs = obs[env.agents[agent_idx]]
            ego_hstate = hstates[agent_idx] # Snapshot state
            policy = policy_pairing.policies[agent_idx]
            
            probs_list = []
            
            # Prepare inputs
            obs_in = jnp.expand_dims(jnp.expand_dims(ego_obs, 0), 0)
            done_in = jnp.zeros((1, 1))
            ac_in = (obs_in, done_in)
            
            for z_idx in range(16):
                 z_onehot = jax.nn.one_hot(jnp.array([z_idx]), 16)
                 
                 if ego_direction:
                     # Fix Ego Z to 0 for consistency
                     z_ego_fixed = jax.nn.one_hot(jnp.array([0]), 16)
                     pp_an = jnp.concatenate([z_onehot, z_ego_fixed], axis=-1)
                 else:
                     pp_an = z_onehot
                     
                 # (Dim) -> (1, 1, Dim)
                 pp_an_in = pp_an[jnp.newaxis, jnp.newaxis, ...]
                 
                 # Forward pass (without updating state)
                 _, pi, _ = policy.network.apply(
                     policy.params,
                     ego_hstate, # Use snapshot
                     ac_in,
                     partner_prediction=pp_an_in
                 )
                 
                 probs = jax.nn.softmax(pi.logits[0, 0])
                 probs_list.append(probs)

            # Statistics
            all_probs = jnp.stack(probs_list) # (16, 6)
            mean_probs = jnp.mean(all_probs, axis=0)
            
            max_diff_vec = jnp.max(all_probs, axis=0) - jnp.min(all_probs, axis=0)
            max_diff_val = jnp.max(max_diff_vec) 
            
            mad_vec = jnp.mean(jnp.abs(all_probs - mean_probs), axis=0)
            avg_mad = jnp.mean(mad_vec)
            
            # Store Result row
            row = {
                "state_step": step,
                "avg_mad": float(avg_mad),
                "max_diff": float(max_diff_val),
                "probs_per_latent": np.array(all_probs)
            }
            results.append(row)
            
        # 3. Env Step
        rng, step_key = jax.random.split(rng)
        obs, env_state, _, done, _ = env.step(step_key, env_state, actions)
        hstates = next_hstates # Update H-states

        # Update history buffers
        for i in range(num_agents):
            agent_id = env.agents[i]
            hist = obs_history[agent_id]
            hist = jnp.roll(hist, shift=-1, axis=0)
            hist = hist.at[-1].set(obs[agent_id])
            obs_history[agent_id] = hist

        if num_agents == 2:
            act_history["agent_0"] = jnp.roll(act_history["agent_0"], shift=-1, axis=0)
            act_history["agent_0"] = act_history["agent_0"].at[-1].set(actions[env.agents[1]])
            act_history["agent_1"] = jnp.roll(act_history["agent_1"], shift=-1, axis=0)
            act_history["agent_1"] = act_history["agent_1"].at[-1].set(actions[env.agents[0]])

    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Header
        header = ["State_Step", "Avg_MAD", "Max_Diff"]
        for z in range(16):
            header.append(f"Latent_{z}_Probs")
        writer.writerow(header)
        
        for row in results:
            line = [
                row["state_step"],
                f"{row['avg_mad']:.6f}",
                f"{row['max_diff']:.6f}"
            ]
            for z in range(16):
                # Format probs: [0.1, 0.2, ...]
                p_vec = row["probs_per_latent"][z]
                p_str = "[" + ", ".join([f"{p:.3f}" for p in p_vec]) + "]"
                line.append(p_str)
            writer.writerow(line)
            
    print(f"  [Sensitivity] Results saved to {save_path}")
    return results
