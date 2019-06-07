from deap import base
from deap import algorithms
from deap import tools
from deap import creator

import random
import functools
import numpy as np
import scipy.signal as signal

creator.create(
    'ResponseMismatch', base.Fitness, weights=(1.0,)
)
creator.create(
    'IIR', list, fitness=creator.ResponseMismatch, nbits=int
)
IIR = creator.IIR


def fitsos(sos, nbits):
    """
    Ajusta la ganancia de una etapa de segundo orden en punto fijo
    para que sus coeficientes puedan representarse en 1.(`nbits`-1)

    :param sos: etapa de segundo orden de filtro digital IIR
       en punto fijo.
    :param nbits: cantidad de bits de la representación en
       punto fijo de los coeficientes de la etapa.
    """
    # Computa el límite númerico de la representación entera signada.
    n = 2**(nbits - 1)
    # Busca el coeficiente de máximo valor absoluto.
    c = sos[np.argmax(np.abs(sos[-1] * sos[:-1]))]
    if abs(c) >= n:
        # Escala todos los coeficientes para que el coeficiente
        # de máximo valor absoluto sea -1 en punto fijo.
        sos[:-1] = -n * sos[-1] * sos[:-1] // abs(c)
        # Conserva el factor de escala aplicado.
        sos[-1] = min(-1, -abs(c) // n)
    return sos


def impulse(iir, ts, n):
    """
    Computa la respuesta al impulso de un filtro digital IIR
    en punto fijo.

    El filtro se representa como una secuencia ordenada de
    etapas de segundo orden, cada una representada por los
    coeficientes de su ecuación en diferencias más una
    ganancia: [b0, b1, b2, a1, a2, k]. El coeficiente a0 se
    asume unitario. Etapas de primer orden pueden obtenerse
    haciendo b2 = a2 = 0.

    :param iir: filtro digital IIR en punto fijo.
    :param ts: período de muestreo, en segundos.
    :param n: cantidad de muestras.
    :return: tiempo y respuesta al impulso, como tupla.
    """
    # Inicializa en cero los vectores de salida de
    # cada una de las M etapas del filtro más el de
    # entrada. Cada vector puede contener N muestras
    # más 2 muestras adicionales para conformar la
    # línea de demora.
    y = np.zeros([len(iir) + 1, n + 2], dtype=int)
    # Inicializar el vector de entrada con un delta
    # discreto.
    y[0, 2] = 2**(iir.nbits-1) - 1
    # Computar para cada instante discreto las salidas
    # de cada etapa, desde la primera hasta la última
    # en ese orden.
    for j in range(2, n + 2):
        for i, sos in enumerate(iir, start=1):
            b0, b1, b2, a1, a2, k = sos
            # Computar la ecuación de diferencias, truncando
            # y saturando el resultado para ser representado
            # en punto fijo 1.(`nbits`-1)
            y[i, j] = np.clip((k * (
                b0 * y[i - 1, j] +
                b1 * y[i - 1, j - 1] +
                b2 * y[i - 1, j - 2] -
                a1 * y[i, j - 1] -
                a2 * y[i, j - 2]
            )) >> (iir.nbits - 1), -2**(iir.nbits - 1), 2**(iir.nbits - 1) - 1)
    # Retorna respuesta al impulso renormalizando la salida
    # al intervalo [-1, 1)
    # salida del
    t = ts * np.arange(n)
    im = y[-1, 2:] / 2**(iir.nbits - 1)
    return t, im


def iir2sos(iir):
    """
    Convierte un filtro digital IIR en punto fijo a su representación
    como secuencia de secciones de segundo orden en punto flotante (ver
    `scipy.signal.sos2tf` como referencia).

    :param iir: filtro digital IIR en punto fijo.
    :return: filtro digital en representación SOS.
    """
    # Computa el límite númerico de la representación entera signada.
    n = 2**(iir.nbits - 1)
    # Escala el filtro digital en punto fijo acorde a la ganancia y
    # normaliza al intervalo [-1, 1) en punto flotante.
    return np.array([
        (*(sos[-1] * sos[:3] / n), 1.,
         *(sos[-1] * sos[3:5] / n))
        for sos in iir
    ])


def genStablePrototype(nlimit, nbits=32):
    """
    Genera un filtro digital IIR en punto fijo estable en
    forma aleatoria.

    :param nlimit: orden máximo admitido para el filtro.
    :param nbits: cantidad de bits utilizados para la
      representación numérica de los coeficientes.
    :return: filtro digital IIR en punto fijo generado.
    """
    iir = IIR()
    # Computa el límite númerico de la representación entera signada.
    n = 2 ** (nbits - 1)
    # Selecciona el orden del filtro en forma aleatoria
    # del intervalo [1, nlimit].
    order = max(int(random.random() * (nlimit + 1)), 1)
    # Si el orden es impar se introduce una etapa de primer orden.
    if order % 2 != 0:
        # Cero y polo de la etapa se ubican dentro o sobre el
        # círculo unidad.
        b0 = n
        b1 = np.random.randint(-n, n-1)
        a1 = np.random.randint(-n, n-1)
        sos = np.array([b0, b1, 0, a1, 0, 1])
        # Ajusta la ganancia de la sección para su representación.
        fitsos(sos, nbits)
        # Incorpora la etapa al filtro.
        iir.append(sos)
    # Introduce N etapas de segundo orden para alcanzar
    # el orden seleccionado.
    for _ in range(order // 2):
        # Ceros y polos de la etapa se ubican dentro del círculo unidad.
        b0 = n
        b2 = np.random.randint(-n+1, n-1)
        a2 = np.random.randint(-n+1, n-1)
        b1 = np.random.randint(-b2-n, b2+n)
        a1 = np.random.randint(-a2-n, a2+n)
        sos = np.array([b0, b1, b2, a1, a2, 1])
        # Ajusta la ganancia de la sección para su representación.
        fitsos(sos, nbits)
        # Incorpora la etapa al filtro.
        iir.append(sos)
    if hasattr(iir, 'nbits'):
        # Preserva el número de bits en el filtro.
        iir.nbits = nbits
    return iir


def cxUniformND(iir1, iir2, ndpb):
    """
    Cruza numeradores y denominadores de filtros digitales IIR en
    punto fijo, potencialmente de distinto orden, produciendo dos
    descendientes. El orden de las etapas a cruzar es modificado
    aleatoriamente. Variante de `deap.tools.cxUniform`.

    :param iir1: primer filtro progenitor.
    :param iir2: segundo filtro progenitor.
    :param ndpb: probabilidad de cruza de numerador y/o denominador.
    """
    # Tomando el filtro candidato de menor orden, itera las
    # secciones de a pares tomados en forma aleatoria.
    for i, j in zip(
        random.sample(list(range(len(iir1))), len(iir1)),
        random.sample(list(range(len(iir2))), len(iir2))
    ):
        # Obtiene las etapas de cada filtro a cruzar.
        sos1 = iir1[i]
        sos2 = iir2[j]
        if random.random() < ndpb:
            # Cruza los numeradores de las etapas
            sos1[:3], sos2[:3] = sos2[:3], sos1[:3]
        if random.random() < ndpb:
            # Cruza los denominadores de las etapas
            sos1[3:5], sos2[3:5] = sos2[3:5], sos1[3:5]
        # Ajusta la ganancia de la primera sección para que los
        # coeficientes del filtro puedan representarse en punto
        # fijo para el número de bits del filtro candidato.
        fitsos(sos1, iir1.nbits)
        # Ajusta la ganancia de la primera sección para que los
        # coeficientes del filtro puedan representarse en punto
        # fijo para el número de bits del filtro candidato.
        fitsos(sos2, iir2.nbits)
    return iir1, iir2


def evTimeResponse(iir, target, ts):
    """
    Evalúa la aptitud de la respuesta temporal de un filtro
    digital IIR en punto fijo según la similitud que su
    respuesta temporal presenta respecto a la respuesta
    objetivo.

    :param iir: filtro digital IIR en punto fijo.
    :param target: respuesta al impulso objetivo.
    :param ts: período de muestreo, en segundos.
    :return: aptitud del filtro provisto.
    """
    # Computa la respuesta al impulso del filtro candidato
    # en su representación SOS.
    _, (im,) = signal.dimpulse(
        (*signal.sos2tf(iir2sos(iir)), ts), n=len(target)
    )
    # Computa el error relativo entre respuesta al impulso
    # del filtro candidato y respuesta al impulso esperada.
    et = (im - target) / np.max(np.abs(target))
    # Evalua la aptitud del filtro candidato como el recíproco
    # de la potencia de error relativo.
    return (1. / (np.mean(et)**2 + np.var(et)),)


def mutCoeffGaussian(iir, mu, sigma, indpb):
    """
    Muta los coeficientes de un filtro digital IIR en punto
    fijo mediante perturbaciones numéricas. Variante de
    `deap.tools.mutGaussian`.

    :param mu: media de la distribución gaussiana de la que se toman
       las perturbaciones a aplicar.
    :param sigma: desvío estandar de la distribución gaussiana de la
       que se toman las perturbaciones a aplicar.
    :param indpb: probabilidad de perturbar un coeficiente.
    """
    # Itera cada sección del filtro.
    for sos in iir:
        # Conforma una máscara de los coeficientes de la
        # sección actual del filtro, según la probabilidad
        # dada.
        mask = (np.random.random(len(sos)-1) < indpb)
        # Perturba los coeficientes a partir de una distribución
        # normal con media y desvío estándar dados.
        sos[:-1][mask] += np.random.normal(
            mu, sigma, np.count_nonzero(mask)
        ).astype(int)
        # Ajusta la ganancia de la sección para que los coeficientes
        # del filtro puedan representarse en punto fijo para el
        # número de bits del filtro.
        fitsos(sos, iir.nbits)
    return iir,


def eaSimplePlusElitism(population, toolbox, cxpb, mutpb, eprop, ngen,
                        stats=None, halloffame=None, verbose=__debug__):
    """
    Variante de `deap.algorithms.eaSimple` con una proporción de elitismo.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evalua los individuos con aptitud inválida.
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Comienza el proceso evolutivo.
    for gen in range(1, ngen + 1):
        # Seleccina la próxima generación de individuos.
        offspring = toolbox.select(population, len(population))

        # Varia el pool de individuos, aplicando cruza y mutación.
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evalua los individuos con aptitud inválida.
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Actualiza el grupo de mejores individuos.
        if halloffame is not None:
            halloffame.update(offspring)

        # Reemplaza la población actual con los mejores del conjunto
        # compuesta por su descendencia y la elite.
        elite_count = int(len(population) * eprop)
        elite = tools.selBest(population, elite_count)
        population[:] = tools.selBest(offspring + elite, len(population))

        # Toma nota de las estadísticas de la generación actual.
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def configure_genetic_approx(*, nbits=16, nlimit=8, nsln=10, cxpb=0.7, ndpb=0.5,
                             mutpb=0.2, mutmean=0.0, mutstd=0.3, coeffpb=0.1,
                             tournsize=3, nind=1000, eprop=0.005, ngen=400,
                             verbose=__debug__):
    """
    Configura una función de aproximación de filtros digitales IIR en punto
    fijo con una dada respuesta al impulso.

    :param nbits: cantidad de bits utilizados para la representación numérica
      de los coeficientes la solución.
    :param nlimit: orden máximo admitido para la solución.
    :param nsln: cantidad de soluciones a conservar de entre las más aptas.
    :param cxpb: probabilidad de cruza genética de soluciones.
    :param ndpb: probabilidad de intercambio de numerador y denominador
      de soluciones seleccionadas para la cruza.
    :param mutpb: probabilidad de mutar una solución.
    :param mutmean: media de perturbaciones utilizadas para la mutación,
      en el intervalo [-1, 1).
    :param mutstd: desvío estándar de las perturbaciones utilizadas para
      la mutación, en el intervalo [-1, 1).
    :param coeffpb: probabilidad de perturbación de un coeficiente de
      una solución seleccionada para la mutación.
    :param tournsize: cantidad de soluciones a someter en cada instancia
      de torneo de selección.
    :param nind: cantidad de soluciones en el pool genético a explorar.
    :param eprop: proporción de elitismo o la cantidad de soluciones de elite
      respecto al total de soluciones en el pool.
    :param ngen: cantidad de generaciones a evolucionar.
    :return: función de aproximación de filtro digital IIR en punto fijo.
    """
    def approx(target, ts, nlimit=nlimit, nsln=nsln):
        """
        Aproxima un filtro digital IIR en punto fijo para que presente la
        respuesta al impulso objetivo.

        :param target: respuesta al impulso objetivo.
        :param ts: período de muestreo, en segundos.
        :param nlimit: orden máximo admitido para la solución.
        :param nsln: cantidad de soluciones a conservar de entre las más aptas.
        :return: filtro digital IIR en punto fijo.
        """
        toolbox = base.Toolbox()
        toolbox.register(
            'individual', genStablePrototype, nlimit=nlimit, nbits=nbits
        )
        toolbox.register(
            'population', tools.initRepeat, list, toolbox.individual
        )
        toolbox.register('mate', cxUniformND, ndpb=ndpb)
        toolbox.register('select', tools.selTournament, tournsize=tournsize)
        toolbox.register(
            'mutate', mutCoeffGaussian, mu=mutmean*2**(nbits-1),
            sigma=mutstd*2**(nbits-1), indpb=coeffpb
        )
        toolbox.register('evaluate', evTimeResponse, target=target, ts=ts)

        stats = tools.Statistics(
            lambda individual: individual.fitness.values
        )
        stats.register('mean_fitness', np.mean)
        stats.register('fitness_stddev', np.std)
        stats.register('min_fitness', np.min)
        stats.register('max_fitness', np.max)

        hall = tools.HallOfFame(
            maxsize=nsln, similar=lambda x, y: (
                np.all(np.equal(np.shape(x), np.shape(y)))
                and np.all(np.equal(x, y))
            )
        )

        population = toolbox.population(nind)
        offspring, logbook = eaSimplePlusElitism(
            population, toolbox, cxpb=cxpb, mutpb=mutpb,
            eprop=eprop, ngen=ngen, stats=stats, halloffame=hall,
            verbose=verbose
        )

        return hall, offspring, logbook
    return approx


if __name__ == '__main__':
    n = 40
    fs = 1000
    ts = 1 / fs
    b_t, a_t = signal.iirdesign(
        wp=0.3, ws=0.6, gpass=1, gstop=40,
        ftype='butter', analog=False
    )
    w, h_t = signal.freqz(b_t, a_t)
    t, (im_t,) = signal.dimpulse((b_t, a_t, ts), n=n)

    approx = configure_genetic_approx(nbits=16, nlimit=8)
    best, pool, logbook = approx(im_t, ts)

    iir_min_err = best[0]
    print('Minimum error', iir_min_err, len(iir_min_err))
    sos_min_err = iir2sos(iir_min_err)
    _, h_min_err = signal.sosfreqz(sos_min_err, worN=w)
    _, im_min_err = impulse(iir_min_err, ts, n)

    iir_min_n = sorted(best, key=lambda iir: len(iir))[0]
    print('Minimum order', iir_min_n, len(iir_min_n))
    sos_min_n = iir2sos(iir_min_n)
    _, h_min_n = signal.sosfreqz(sos_min_n, worN=w)
    _, im_min_n = impulse(iir_min_n, ts, n)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.plot(w, 20 * np.log10(abs(h_min_n)))
    ax.plot(w, 20 * np.log10(abs(h_min_err)))
    ax.plot(w, 20 * np.log10(abs(h_t)))
    ax.set_xlabel('Frequency [radians / sample]')
    ax.set_ylabel('Amplitude [dB]')
    ax.grid(which='both', axis='both')

    ax = fig.add_subplot(312)
    ax.plot(w, np.angle(h_min_n))
    ax.plot(w, np.angle(h_min_err))
    ax.plot(w, np.angle(h_t))
    ax.set_xlabel('Frequency [radians / sample]')
    ax.set_ylabel('Angle [radians]')
    ax.grid(which='both', axis='both')

    ax = fig.add_subplot(313)
    ax.plot(t, im_min_n)
    ax.plot(t, im_min_err)
    ax.plot(t, im_t)
    ax.set_xlabel('Time [seconds]')
    ax.set_ylabel('Amplitude [ ]')
    ax.grid(which='both', axis='both')

    zeros_min_n, poles_min_n, gain_min_n = signal.sos2zpk(sos_min_n)
    zeros_min_err, poles_min_err, gain_min_err = signal.sos2zpk(sos_min_err)
    zeros_o, poles_o, gains_o = [], [], []
    for iir in pool:
        z, p, k = signal.sos2zpk(iir2sos(iir))
        zeros_o.extend(z); poles_o.extend(p); gains_o.append(k)
    zeros_t, poles_t, gain_t = signal.tf2zpk(b_t, a_t)

    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.scatter(x=np.real(zeros_t), y=np.imag(zeros_t), color='blue')
    ax.scatter(x=np.real(zeros_min_err), y=np.imag(zeros_min_err), color='red')
    ax.scatter(x=np.real(zeros_min_n), y=np.imag(zeros_min_n), color='green')
    ax.scatter(x=np.real(zeros_o), y=np.imag(zeros_o), color='yellow')
    ax.axis([-2, 2, -2, 2])
    ax = fig.add_subplot(312)
    ax.scatter(x=np.real(poles_t), y=np.imag(poles_t), marker='x', color='blue')
    ax.scatter(x=np.real(poles_o), y=np.imag(poles_o), color='yellow')
    ax.scatter(x=np.real(poles_min_err), y=np.imag(poles_min_err), marker='x', color='red')
    ax.scatter(x=np.real(poles_min_n), y=np.imag(poles_min_n), marker='x', color='green')
    ax.axis([-2, 2, -2, 2])
    ax = fig.add_subplot(313)
    ax.hist(gains_o, bins=100, density=True, color='yellow')
    ax.axvline(x=gain_t, color='blue')
    ax.axvline(x=gain_min_err, color='red')
    ax.axvline(x=gain_min_n, color='green')
    plt.show()
