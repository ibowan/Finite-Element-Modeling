import math
E, H, sig0, A, L = 210e3, 10e3, 50.0, 1.0, 1.0
le = [L/2]*2
states = [{'ep':0.0,'a':0.0} for _ in range(2)]
def law(eps, st, plastic=True):
    if not plastic:
        sig = E*eps
        return sig, E, st['ep'], st['a']
    ep, a = st['ep'], st['a']
    sig_t = E*(eps-ep)
    f = abs(sig_t) - (sig0 + H*a)
    if f <= 0:
        return sig_t, E, ep, a
    dg = f/(E+H)
    s = 1.0 if sig_t >=0 else -1.0
    sig = sig_t - E*dg*s
    return sig, (E*H)/(E+H), ep+dg*s, a+dg
u1 = 0.0
hist = []
nsteps, u_max = 200, 0.02
du = u_max/nsteps
for step in range(nsteps+1):
    uR = step*du
    for _ in range(25):
        eps1, eps2 = (u1-0)/le[0], (uR-u1)/le[1]
        sig1,t1,ep1,a1 = law(eps1, states[0], False)
        sig2,t2,ep2,a2 = law(eps2, states[1])
        res = A*(sig1 - sig2)
        if abs(res) < 1e-6:
            break
        kt = A*(t1/le[0] + t2/le[1])
        u1 -= res/kt
    states = [{'ep':ep1,'a':a1},{'ep':ep2,'a':a2}]
    hist.append((uR,u1,sig1,sig2))
for r in hist:
    print('uR=%.5f uM=%.5f sig1=%.1f sig2=%.1f'%r)
