import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
import copy

from bay_ms_utils import * 
from data_enhancement import data_enhancement

def train_spread(model, args, train_dataloader, val_dataloader = None):

    model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    betas = cosine_beta_schedule(args.timesteps)
    alphas = 1 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0).to(args.device)

    best_val_loss = np.inf
    tol_violation_epoch = args.patience
    cur_violation_epoch = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0

        for indx_batch, (batch, obj_values) in enumerate(train_dataloader):
            optimizer.zero_grad()

            # Extract batch points
            points = batch.to(args.device)
            points.requires_grad = True  # Enable gradients if needed
            obj_values = obj_values.to(args.device)

            # Sample random timesteps for each data point
            t = torch.randint(0, args.timesteps, (points.shape[0],)).to(args.device)
            alpha_bar_t = alpha_cumprod[t].unsqueeze(1)  # shape: [batch_size, 1]
            
            # Forward process: Add noise to points
            noise = torch.randn_like(points).to(
                    args.device
                )  # shape: [batch_size, input_dim]
            x_t = (
                    torch.sqrt(alpha_bar_t) * points
                    + torch.sqrt(1 - alpha_bar_t) * noise
                )

            # Conditioning information
            c = obj_values.detach()
            Delta_c = c[c > 0].min() if (c > 0).any() else 1e-6
            c = c + Delta_c
            # Model predicts noise
            predicted_noise = model(
                    x_t, t.float() / args.timesteps, c
                )

            # Compute loss
            ## DDPM loss
            loss_simple = l_simple_loss(predicted_noise, noise)

            loss_simple.backward()
            optimizer.step()
            epoch_loss += loss_simple.item()

        epoch_loss = epoch_loss / len(train_dataloader)
            
        if args.patience < np.inf:
            assert val_dataloader is not None, "Validation dataloader is required for early stopping."
            # Validation
            model.eval()
            val_loss = 0.0
            for indx_batch, (val_batch, val_obj_values) in enumerate(val_dataloader):
                # Extract batch points
                val_points = val_batch.to(
                    args.device
                )  # TensorDataset wraps data in a tuple
                val_points.requires_grad = True  # Enable gradients if needed
                val_obj_values = val_obj_values.to(args.device)

                # Sample random timesteps for each data point
                t = torch.randint(0, args.timesteps, (val_points.shape[0],)).to(args.device)
                alpha_bar_t = alpha_cumprod[t].unsqueeze(1)  # shape: [batch_size, 1]

                # Forward process: Add noise to points
                val_noise = torch.randn_like(val_points).to(
                    args.device
                )  # shape: [batch_size, input_dim]
                val_x_t = (
                    torch.sqrt(alpha_bar_t) * val_points
                    + torch.sqrt(1 - alpha_bar_t) * val_noise
                )

                # Conditioning information
                c = val_obj_values.detach()
                Delta_c = c[c > 0].min() if (c > 0).any() else 1e-6
                c = c + Delta_c
                # Model predicts noise
                val_predicted_noise = model(
                    val_x_t, t.float() / args.timesteps, c
                )
                loss_simple = l_simple_loss(val_predicted_noise, val_noise)

                val_loss += loss_simple.item()

            val_loss = val_loss / len(val_dataloader)

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                cur_violation_epoch = 0
            else:
                cur_violation_epoch += 1
                if cur_violation_epoch >= tol_violation_epoch:
                    break
    
    return model


def one_sampling_step_spread(
    model, p_model, x_t, t, 
    beta_t, alpha_bar_t, args, surrogate_model, prev_pf_points
):

    # Create a tensor of timesteps with shape (num_points_sample, 1)
    t_tensor = torch.full(
        (x_t.shape[0],),
        t,
        device=args.device,
        dtype=torch.float32,
    )
    # Compute objectibe values
    obj_values = torch.cat(
        objective_functions_GP(x_t, surrogate_model, args.coef_lcb, args.device), dim=1
    )  # shape: [num_points_sample, num_objectives]
    c = obj_values.detach()
    with torch.no_grad():
        predicted_noise = model(x_t, 
                                t_tensor / args.timesteps, 
                                c)

    #### Reverse diffusion step
    sqrt_1_minus_alpha_t = torch.sqrt(torch.clamp(1 - alpha_bar_t, min=1e-6))
    sqrt_1_minus_beta_t = torch.sqrt(torch.clamp(1 - beta_t, min=1e-6))
    mean = (1 / sqrt_1_minus_beta_t) * (
        x_t - (beta_t / sqrt_1_minus_alpha_t) * (predicted_noise)
    )
    if torch.isnan(mean).any():
        print("Nan values in mean !")
    std_dev = torch.sqrt(beta_t)
    z = torch.randn_like(x_t) if t > 0 else 0.0  # No noise for the final step
    x_t = mean + std_dev * z

    #### Pareto Guidance step
    if args.num_obj == 2:
        x_t.data = repair_bounds(x_t.data.clone(), args.bounds[0], args.bounds[1])
        X = x_t.clone().detach().requires_grad_()
        _, list_grad_i = objective_functions_GP(
            X, surrogate_model, args.coef_lcb, args.device, get_grad=True
            )
        # X.grad.zero_()
        grad_f1, grad_f2 = list_grad_i[0], list_grad_i[1]
        # Normalize gradients
        grad_f1 = torch.nn.functional.normalize(grad_f1, dim=0)
        grad_f2 = torch.nn.functional.normalize(grad_f2, dim=0)
        if torch.isnan(grad_f1).any():
            print("Nan values in grad_f1 bsolveh !")
        if torch.isnan(grad_f2).any():
            print("Nan values in grad_f2 bsolveh !")
        grads = torch.stack([grad_f1, grad_f2], dim=0) # (m, N, d)
        grads_copy = torch.stack([grad_f1, grad_f2], dim=1).detach() # (N, m, d)
        # Compute the MGD gradient
        g_x_t_minus = solve_min_norm_2_loss(grad_f1, grad_f2)
        
    else:
        x_t.data = repair_bounds(x_t.data.clone(), args.bounds[0], args.bounds[1])
        X = x_t.clone().detach().requires_grad_()
        _, list_grad_i = objective_functions_GP(
            X, surrogate_model, args.coef_lcb, args.device, get_grad=True
            )
        for i in range(len(list_grad_i)):
            # X.grad.zero_()
            grad_i = list_grad_i[i]
            # Normalize gradients
            grad_i = torch.nn.functional.normalize(grad_i, dim=0)
            if torch.isnan(grad_i).any():
                print("Nan values in grad_i bsolveh !")
            list_grad_i[i] = grad_i
        
        grads = torch.stack(list_grad_i, dim=0) # (m, N, d)
        grads_copy = torch.stack(list_grad_i, dim=1).detach() # (N, m, d)
        # Compute the MGD gradient
        g_x_t_minus = get_mgd_grad(list_grad_i)
        

    if torch.isnan(x_t).any():
        print("Nan values in x_t bsolveh !")
        print("x_t: ", x_t)

    if torch.isnan(g_x_t_minus).any():
        print("Nan values in g_x_t_minus bsolveh !")
        print("g_x_t_minus: ", g_x_t_minus)

    eta = mgd_armijo_step(
        x_t,
        g_x_t_minus,
        obj_values,
        grads_copy,
        objective_functions_GP,
        surrogate_model,
        args,
        eta_init=args.eta,
    )
    h_star, _ = solve_for_h(
        x_t,  # x_t after the reverse diffusion step is x_t_prime
        objective_functions_GP,
        surrogate_model,
        g_x_t_minus,
        grads,
        args,
        eta=eta,
        lambda_rep=args.lambda_rep, 
        sigma=args.kernel_sigma,
        use_sigma=False,
        num_inner_steps=args.num_inner_steps,
        lr_inner=args.lr_inner,
    )

    if torch.isnan(h_star).any():
        print("Nan values in h_star !")
        print("h_star: ", h_star)

    x_t = x_t - eta * h_star
    
    pf_population = repair_bounds(
                    copy.deepcopy(x_t.detach()), args.bounds[0], args.bounds[1], args
                )
    
    if t < args.timesteps - 1:
        assert prev_pf_points is not None, "prev_pf_points should not be None at this step."
    
    if prev_pf_points is not None:
        all_pop_points = torch.cat((prev_pf_points, pf_population), dim=0)       
        pf_points = select_top_n_for_BaySpread(
                        all_pop_points,
                        p_model,
                        args.device,
                        surrogate_model,
                        args.coef_lcb,
                        n = x_t.shape[0]
                    )
    else:
        pf_points = pf_population

    return x_t, pf_points


def gen_offspring_via_spread(model, Parent, surrogate_model, p_model, args):
    
    rows_to_take = int(1 / 3 * Parent.shape[0])
    pop = Parent[:rows_to_take, :]
    if len(pop) % 2 == 1:
        pop = pop[:-1]
    
    augmentation_factor = 10
    augmented_pop = data_enhancement(
            pop,
            augmentation_factor=augmentation_factor
        )

    dataset = torch.tensor(augmented_pop).float().to(args.device)
    
    ##### TRAINING #####
    if args.patience < np.inf:
        # Split dataset: 10% for evaluation, 90% for training
        total_size = len(dataset)
        eval_size = int(0.10 * total_size)
        train_size = total_size - eval_size

        dataset = dataset[torch.randperm(dataset.size(0))]
        train_dataset = dataset[:train_size]
        eval_dataset = dataset[train_size:]
        
        y_train = torch.cat(
            objective_functions_GP(train_dataset, surrogate_model, args.coef_lcb, args.device), 
            dim=1
        )

        y_val = torch.cat(
            objective_functions_GP(eval_dataset, surrogate_model, args.coef_lcb, args.device), 
            dim=1
        )
        dataset_train = torch.utils.data.TensorDataset(train_dataset, y_train)
        dataset_val = torch.utils.data.TensorDataset(eval_dataset, y_val)
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
        )
        eval_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            shuffle=False,
        )
        
    else:
        # Use the entire dataset for training
        y_train = torch.cat(
            objective_functions_GP(dataset, surrogate_model, args.coef_lcb, args.device), dim=1
        )
        dataset_train = torch.utils.data.TensorDataset(dataset, y_train)
        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=True 
        )
        eval_loader = None
        
    model = train_spread(model, args, train_loader, eval_loader)

    new_offsprings = []
    for i in range(args.num_samp_iters):
        betas = cosine_beta_schedule(args.timesteps)
        alphas = 1 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0).to(args.device)
        
        # Start from Gaussian noise
        x_t = torch.rand(dataset.shape) # in [0, 1]
        x_t = x_t.to(args.device)
        x_t.requires_grad = True
        
        prev_pf_points = None
        model.eval()
            
        for t in reversed(range(args.timesteps)):
            x_t.requires_grad_(True)
            x_t.data = repair_bounds(
                    x_t.data.clone(), args.bounds[0], args.bounds[1], args
                )
            # Compute beta_t and alpha_t
            beta_t = 1 - alphas[t]
            alpha_bar_t = alpha_cumprod[t]

            x_t, pf_points = one_sampling_step_spread(
                model,
                p_model,
                x_t,
                t,
                beta_t,
                alpha_bar_t,
                args,
                surrogate_model,
                prev_pf_points
            )
            
            prev_pf_points = pf_points
            
            x_t = x_t.detach()
            
        new_offsprings.append(pf_points.detach().cpu().numpy())

    result = np.vstack(new_offsprings)
    
    return result

            
