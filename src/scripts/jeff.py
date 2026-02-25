import numpy as np 
import swap_errors.tcc_model as stm
import swap_errors.theory as swt

#%%
bounds = (1.5, 5)
snrs = np.linspace(*bounds, 1000)
wid = 1.65

snrs_fit = np.linspace(*bounds, 8)
sig = np.zeros((len(snrs_fit), 500))
guess_rates = np.zeros((len(snrs_fit), 500))
for i, snr in enumerate(snrs_fit):
    m = swt.SimplifiedDiscreteKernelTheory(snr, wid)
    ts, rs = m.simulate_decoding(10000)
    outs = stm.fit_corr_guess_model(ts, rs)["samples"]
    sig[i] = outs["sigma"]
    guess_rates[i] = outs["resp_rate"][:, -1]

f, ax = plt.subplots(1, 1)

ax.plot(np.mean(sig, axis=1), np.mean(guess_rates, axis=1), "-o")
errs = np.sqrt(swt.local_err(snrs, wid))
gs = swt.threshold_prob(snrs, wid)
ax.plot(errs, gs)

ax.set_xlabel(r"local errors $\sigma$") 
ax.set_ylabel("guess rate") 
plt.show()

