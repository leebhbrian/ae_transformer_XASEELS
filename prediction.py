import torch
from autoencoder import DenoisingAutoencoder
from transformer import tfuncondRegressor
import numpy as np
from analysis_util import plot_parity
###Load models
device      = f"cuda" if torch.cuda.is_available() else "cpu"
ae_model= DenoisingAutoencoder(latent_dim=8).to(device)
ae_model.load_state_dict(torch.load(f"./model/autoencoder.pt", map_location=device))
ae_model.eval()
for p in ae_model.parameters():
    p.requires_grad = False
tf_paths = ["./model/tf_v0.pt", "./model/tf_v1.pt", "./model/tf_v2.pt"]
tf_models = []
for path in tf_paths:
    tf_model = tfuncondRegressor(seq_len=451, d_lat=8).to(device)
    tf_model.load_state_dict(torch.load(path, map_location=device))
    tf_model.eval()
    tf_models.append(tf_model)
ae_model.to(device)


def ensemble_predict(inputter, recon, latent):
    """Run the ensemble of transformers and return (mean, std) of the
    oxidation-state prediction (output column 0)."""
    preds = []
    with torch.no_grad():
        for tf_model in tf_models:
            pred = tf_model(inputter, recon, latent)
            preds.append(pred.detach().cpu().numpy()[0, 0])
    preds = np.array(preds)
    return preds.mean(), preds.std()

###Literature experimental XAS prediction
print("XAS prediction")
xas_list=['Cu','Cu2O','CuO','ZrCuSiAs', 'LaCuSeO', 'LaCuTeO', 'ZrCuSiP', 'LaCuSO', 'CuFeS2']
ox_list=[0,1,2,1,1,1,1,1,1,1,1]
res=[]
for i1 in range(len(xas_list)):
    matnow=xas_list[i1]
    oxnow=ox_list[i1]
    snow=np.load(f'./data/experimental_XAS_lit/{matnow}.npy')
    specs_t = torch.as_tensor(snow, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    recon, latent = ae_model(specs_t)
    recon  = recon.squeeze(1)
    latent = latent
    inputter=specs_t.squeeze(1)
    predox, predstd = ensemble_predict(inputter, recon, latent)
    res.append([oxnow, predox, predstd])
    print(f'{matnow} done: {predox} +/- {predstd}, target: {oxnow}')
plot_parity(np.array(res), './prediction_result/xas_parity.png')
print()

###Mixed experimental XAS prediction
print("Mixed XAS prediction")
num_al=40
resol=0.05
snow0=np.load(f'./data/experimental_XAS_lit/Cu.npy')
snow1=np.load(f'./data/experimental_XAS_lit/Cu2O.npy')
snow2=np.load(f'./data/experimental_XAS_lit/CuO.npy')
res_xas_mix=[]
for i in range(num_al):
    if i<=20:    
        snow=snow0*float(1.0-resol*i)+snow1*float(resol*i)
        snow=(snow-np.min(snow))/(np.max(snow)-np.min(snow))
        oxnow=i*resol
    else:
        snow=snow1*float(1.0-resol*(i-20))+snow2*float(resol*(i-20))
        snow=(snow-np.min(snow))/(np.max(snow)-np.min(snow))
        oxnow=(i)*resol
    specs_t = torch.as_tensor(snow, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    recon, latent = ae_model(specs_t)
    recon  = recon.squeeze(1)
    latent = latent
    inputter=specs_t.squeeze(1)
    predox, predstd = ensemble_predict(inputter, recon, latent)
    res_xas_mix.append([oxnow, predox, predstd])
    print(f"{oxnow},{predox},{predstd}")
res_xas_mix=np.array(res_xas_mix)
plot_parity(np.array(res_xas_mix), './prediction_result/xas_mix.png')
print()

###Mixed experimental XAS prediction
print("Mixed EELS prediction")
num_al=40
resol=0.05
snow0=np.load(f'./data/experimental_EELS/cu0.npy')
snow1=np.load(f'./data/experimental_EELS/cu1.npy')
snow2=np.load(f'./data/experimental_EELS/cu2.npy')
res_EELS=[]
for i in range(num_al):
    if i<=20:    
        snow=snow0*float(1.0-resol*i)+snow1*float(resol*i)
        snow=(snow-np.min(snow))/(np.max(snow)-np.min(snow))
        oxnow=i*resol
    else:
        snow=snow1*float(1.0-resol*(i-20))+snow2*float(resol*(i-20))
        snow=(snow-np.min(snow))/(np.max(snow)-np.min(snow))
        oxnow=(i)*resol
    specs_t = torch.as_tensor(snow, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    recon, latent = ae_model(specs_t)
    recon  = recon.squeeze(1)
    latent = latent
    inputter=specs_t.squeeze(1)
    predox, predstd = ensemble_predict(inputter, recon, latent)
    res_EELS.append([oxnow, predox, predstd])
    print(f"{oxnow},{predox},{predstd}")
res_EELS=np.array(res_EELS)
plot_parity(np.array(res_EELS), './prediction_result/EELS_mix.png')