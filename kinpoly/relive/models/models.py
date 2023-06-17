
def get_model(cfg, device, mjdict, data, dataset):
    if cfg.model_name == 'ar':
        model = KinMLP(cfg, device, mjdict, data)
    elif cfg.model_name == 'reg':
        model = KinReg(cfg, device, mjdict, data)
    elif cfg.model_name == 'sim':
        model = KinSim(cfg, device, mjdict, data)
    elif cfg.model_name == 'refine':
        model = KinRefine(cfg, device, mjdict, data, dataset)
    return model