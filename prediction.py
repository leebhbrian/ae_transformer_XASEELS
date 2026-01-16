import torch
from autoencoder import DenoisingAutoencoder
from transformer import tfuncondRegressor
import numpy as np
from analysis_util import plot_parity
###Load models
device      = f"cuda" if torch.cuda.is_available() else "cpu"
ae_model= DenoisingAutoencoder(latent_dim=16).to(device)
ae_model.load_state_dict(torch.load(f"./model/autoencoder.pt"))
ae_model.eval()
for p in ae_model.parameters():
    p.requires_grad = False
tf_model = tfuncondRegressor(seq_len=451, d_lat=16).to(device)
tf_model.load_state_dict(torch.load(f"./model/transformer.pt"))
tf_model.eval()
ae_model.to(device)
tf_model.to(device)

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
    with torch.no_grad():
        pred = tf_model(inputter,recon, latent)
    predox=pred.detach().cpu().numpy()[0,0]
    res.append([oxnow, predox])
    print(f'{matnow} done: {predox}, target: {oxnow}')
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
    with torch.no_grad():
        pred = tf_model(inputter,recon, latent)
    predox=pred.detach().cpu().numpy()[0,0]
    res_xas_mix.append([oxnow, predox])
    print(f"{oxnow},{predox}")
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
    with torch.no_grad():
        pred = tf_model(inputter,recon, latent)
    predox=pred.detach().cpu().numpy()[0,0]
    res_EELS.append([oxnow, predox])
    print(f"{oxnow},{predox}")
res_EELS=np.array(res_EELS)
plot_parity(np.array(res_EELS), './prediction_result/EELS_mix.png')