import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.realpath(__file__)
)))

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

import fwliir

n = 40
fs = 44000
ts = 1 / fs
b_t, a_t = signal.iirdesign(
    wp=0.3, ws=0.6, gpass=1, gstop=40,
    ftype='butter', analog=False
)

w, h_t = signal.freqz(b_t, a_t, fs=fs)
t, (im_t,) = signal.dimpulse((b_t, a_t, ts), n=n)

approx = fwliir.configure_genetic_approx(
    nbits=16, nlimit=10, poolsize=500, ngen=500, verbose=True
)
best_slns, pool, logbook = approx(im_t, ts)

iir_min_err = best_slns[0]
sos_min_err = fwliir.iir2sos(iir_min_err)
_, h_min_err = signal.sosfreqz(sos_min_err, worN=w)
_, im_min_err = fwliir.impulse(iir_min_err, ts, n)

iir_min_n = sorted(best_slns, key=lambda iir: len(iir))[0]
sos_min_n = fwliir.iir2sos(iir_min_n)
_, h_min_n = signal.sosfreqz(sos_min_n, worN=w)
_, im_min_n = fwliir.impulse(iir_min_n, ts, n)

fig = plt.figure()
ax = fig.add_subplot(311)
ax.plot(w, 20 * np.log10(abs(h_min_n)))
ax.plot(w, 20 * np.log10(abs(h_min_err)))
ax.plot(w, 20 * np.log10(abs(h_t)))
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Amplitud [dB]')
ax.grid(which='both', axis='both')

ax = fig.add_subplot(312)
ax.plot(w, np.angle(h_min_n))
ax.plot(w, np.angle(h_min_err))
ax.plot(w, np.angle(h_t))
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Fase [rad]')
ax.grid(which='both', axis='both')

ax = fig.add_subplot(313)
ax.plot(t, im_min_n)
ax.plot(t, im_min_err)
ax.plot(t, im_t)
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Amplitud [1]')
ax.grid(which='both', axis='both')

zeros_o, poles_o, gains_o = [], [], []
for iir in pool:
    z, p, k = signal.sos2zpk(fwliir.iir2sos(iir))
    zeros_o.extend(z); poles_o.extend(p); gains_o.append(k)
zeros_t, poles_t, gain_t = signal.tf2zpk(b_t, a_t)
zeros_min_n, poles_min_n, gain_min_n = signal.sos2zpk(sos_min_n)
zeros_min_err, poles_min_err, gain_min_err = signal.sos2zpk(sos_min_err)

fig = plt.figure()
ax = fig.add_subplot(311)
# ax.scatter(x=np.real(zeros_o), y=np.imag(zeros_o), color='yellow')
ax.scatter(x=np.real(zeros_t), y=np.imag(zeros_t), color='blue')
ax.scatter(x=np.real(zeros_min_err), y=np.imag(zeros_min_err), color='red')
ax.scatter(x=np.real(zeros_min_n), y=np.imag(zeros_min_n), color='green')
ax.axis([-2, 2, -2, 2])
ax = fig.add_subplot(312)
# ax.scatter(x=np.real(poles_o), y=np.imag(poles_o), color='yellow')
ax.scatter(x=np.real(poles_t), y=np.imag(poles_t), marker='x', color='blue')
ax.scatter(x=np.real(poles_min_err), y=np.imag(poles_min_err), marker='x', color='red')
ax.scatter(x=np.real(poles_min_n), y=np.imag(poles_min_n), marker='x', color='green')
ax.axis([-2, 2, -2, 2])
ax = fig.add_subplot(313)
ax.hist(gains_o, bins=100, density=True, color='yellow')
ax.axvline(x=gain_t, color='blue')
ax.axvline(x=gain_min_err, color='red')
ax.axvline(x=gain_min_n, color='green')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy(1. / np.array(logbook.select('max_fitness')), 'red')
ax.semilogy(1. / np.array(logbook.select('mean_fitness')), 'blue')

plt.show()
