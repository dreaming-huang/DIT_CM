def cm_training_losses_mutilStep(self, model,model_target,model_teacher, x_start, t,model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """


        t_target=t-1
        t_target = th.where(t_target < 0, th.zeros_like(t), t_target)

        # print("t is :",t)
        map_tensor = th.tensor(self.timestep_map, device=t.device, dtype=t.dtype)

        
            
        c_skip_start, c_out_start = self.scalings_for_boundary_conditions_timestep(
            map_tensor[t], #timestep_scaling= 10000 #timestep_scaling=args.timestep_scaling_factor
        )
        c_skip_start, c_out_start = [self.append_dims(x, x_start.ndim) for x in [c_skip_start, c_out_start]]

        c_skip, c_out = self.scalings_for_boundary_conditions_timestep(
            map_tensor[t_target], #timestep_scaling= 10000 #timestep_scaling=args.timestep_scaling_factor
        )
        c_skip, c_out = [self.append_dims(x, x_start.ndim) for x in [c_skip, c_out]]


        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}


        model_output = model(x_t, t, **model_kwargs)

        if self.model_var_type in [
            ModelVarType.LEARNED,
            ModelVarType.LEARNED_RANGE,
        ]:
            B, C = x_t.shape[:2]
            assert model_output.shape == (B, C * 2, *x_t.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)


        pred_x_0=self._predict_xstart_from_eps(x_t,t,model_output) #因为αβ已经改了，所以这里直接用t

        model_pred = c_skip_start * x_t + c_out_start * pred_x_0


        with th.no_grad():
            with th.autocast("cuda"):
                #1 计算老师前进一步的结果
                # z = th.cat([z, z], 0)
                y_null = th.tensor([1000] * x_t.shape[0], device=x_t.device)
                # print(model_kwargs)
                y=model_kwargs['y']
                teacher_kwargs=dict(**model_kwargs)
                y = th.cat([y, y_null], 0)
                teacher_kwargs['y']=y
                teacher_kwargs['cfg_scale']=4.0
                
                # x_prev=self.ddim_sample_cm(model_teacher,x_t,t,model_kwargs=model_kwargs)["sample"]

                x_prev=self.ddim_sample_cm(model_teacher,th.cat([x_t,x_t],0),th.cat([t,t],0),model_kwargs=teacher_kwargs)["sample"]
                x_prev, x_prev_nocfg = x_prev.chunk(2, dim=0)  # Remove null class samples

                #2 对该结果进行一次求解
                # print(type(x_prev),type)
                model_target_output,_ = th.split(model_target(x_prev, t_target, **model_kwargs), C, dim=1)

                pred_target_x_0=self._predict_xstart_from_eps(x_prev,t_target,model_target_output)
                model_target_pred = c_skip * x_prev + c_out * pred_target_x_0

        # print(alphas_cumprod.shape)
        # print((model_pred.float() - model_target_pred.float())).shape
        # alphas_cumprod =_extract_into_tensor(self.alphas_cumprod,t,model_pred)
        alphas_cumprod_sqrt=th.tensor(self.alphas_cumprod,device=model_pred.device).sqrt()[t].view(model_pred.size(0), 1, 1, 1)
        # alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        alphas_cumprod_sqrt_prev=th.tensor(self.alphas_cumprod_prev,device=model_pred.device)[t].view(model_pred.size(0), 1, 1, 1)
        # weight= th.clamp(1/alphas_cumprod,1,100)
        weight=1
        eps_conf=(1-alphas_cumprod_sqrt_prev).sqrt()/alphas_cumprod_sqrt_prev.sqrt()
        # terms["loss"] = F.mse_loss(eps_conf*model_output.float(), eps_conf*model_target_output.float(), reduction="mean")
        # terms["loss"] = th.mean(
        #     th.sqrt((model_pred.float() - model_target_pred.float()) ** 2 + 0.001**2) - 0.001
        # ) #+terms["vb"]
        # terms["loss"]=th.mean((model_output.float() - model_target_output.float()))
        terms["loss"] = th.mean(
            weight*th.sqrt((model_pred.float() - model_target_pred.float()) ** 2 + 0.001**2) - 0.001
        )
        # terms["loss"] = th.mean(
        #     weight*th.sqrt((eps_conf*model_output.float() - eps_conf*model_target_output.float()) ** 2 + 0.001**2) - 0.001
        # ) #+terms["vb"]
        # loss = torch.mean(
        #     torch.sqrt((model_pred.float() - model_target_pred.float()) ** 2 + args.huber_c**2) - args.huber_c
        # )


        return terms
