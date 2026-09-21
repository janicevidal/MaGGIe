# from .resnet import res_shortcut_22 # Standard MGM decoder
# from .resnet_fam import res_shortcut_fam_22 # MGM + TCVOM decoder
# from .resnet_inst_matt import res_shortcut_inst_matt_22 # MGM + IMD
# from .resnet_inst_matt_spconv import res_shortcut_inst_matt_spconv_22 # MaGGIe: IMD + Spconv
# from .resnet_inst_matt_spconv_temp import res_shortcut_inst_matt_spconv_temp_22 # MaGGIe_Temp: IMD + Spconv + Temporal
# from .shm import shm # SparseMat
from .biref_decoder import biref_decoder
from .biref_aspp_decoder import biref_aspp_decoder
from .focal_decoder import focal_decoder, FocalDecoder
from .focal_uncertainty_decoder import focal_uncertainty_decoder, FocalUncertaintyDecoder