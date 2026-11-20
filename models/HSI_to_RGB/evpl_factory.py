"""Unified EVPL / opsin-layer factory for MODID and MHSI experiments."""

OPSIN_MODES = ('free', 'opsin_fixed', 'opsin_train')


def _set_lambda_trainable(evpl_module, trainable: bool):
    g = evpl_module.filter_spectral_response
    for name in ('lamda_max_1', 'lamda_max_2', 'lamda_max_3'):
        if hasattr(g, name):
            p = getattr(g, name)
            p.requires_grad = trainable


def build_evpl(dataset: str, opsin_mode: str, device, wavelengths=(600.0, 540.0, 460.0)):
    """
    dataset: 'modid' | 'mhsi'
    opsin_mode:
      - free         : FreeConv 1×1 spectral projection (fully learnable)
      - opsin_fixed  : Gaussian opsin, λ_max frozen
      - opsin_train  : Gaussian opsin, λ_max learnable
    """
    if opsin_mode not in OPSIN_MODES:
        raise ValueError(f"opsin_mode must be one of {OPSIN_MODES}")

    w1, w2, w3 = wavelengths

    if dataset == 'modid':
        from models.HSI_to_RGB.modid_EVPL import EVPL
        from models.HSI_to_RGB.modid_EVPL_ablation import EVPL_FreeConv
        if opsin_mode == 'free':
            return EVPL_FreeConv(device)
        evpl = EVPL(w1, w2, w3, device)
        _set_lambda_trainable(evpl, trainable=(opsin_mode == 'opsin_train'))
        return evpl

    if dataset == 'mhsi':
        from models.HSI_to_RGB.mhsi_EVPL import EVPL
        from models.HSI_to_RGB.mhsi_EVPL_ablation import EVPL_FreeConv
        if opsin_mode == 'free':
            return EVPL_FreeConv(device)
        evpl = EVPL(w1, w2, w3, device)
        _set_lambda_trainable(evpl, trainable=(opsin_mode == 'opsin_train'))
        return evpl

    raise ValueError(f"Unknown dataset '{dataset}'")
