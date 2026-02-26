Orbital Exhaustion is a constructive computational framework for the analytic continuation of functional iteration. The core philosophy of the engine is to exhaust all data and complexity of global iteration by decomposing it into discrete orbital mechanics and local analytic propagation. This repository implements numerical methods for solving such transcendental functional equations, currently focusing on parabolic tetration.

Traditional methods often struggle with the functional equation $f(z+1)={\Upsilon}(f(z))=e^{\frac{f(z)}{e}}$ near the parabolic fixed point $z=e$ where ${\Upsilon}'(e)=1$. This engine, however,  exploits that if the recursion $f(z+1)={\Upsilon}\circ{f(z)}$ holds globally for all $z$, then the following:

$$P(z)=\prod_{t=a}^{z-1}{\Upsilon}'\circ{f(t)}=\frac{f'(z)}{f'(a)}$$
$$\frac{P'(z)}{P(z)}=\frac{P'(z+n)}{P(z+n)}-f'(a)\sum_{t=0}^{n-1}{\frac{{\Upsilon}''\circ{f(z+t)}}{{\Upsilon}'\circ{f(z+t)}}P(z+t)}$$

If for the given initial condition, the orbit of $z$ in $f$ converges to a fixed point, then under the limit $\lim_{n\to\infty}$, the following:

$$P'(z)=\ln({\lambda})P(z)-f'(a)P(z)\sum_{t=0}^{\infty}{\frac{{\Upsilon}''\circ{f(z+t)}}{{\Upsilon}'\circ{f(z+t)}}P(z+t)}$$

Where ${\lambda}$ is the multiplier given by ${\Upsilon}'(z_{\star})$ with $z_{\star}$ the fixed point such that ${\Upsilon}(z_{\star})=z_{\star}$. Then for ${\Upsilon}(z)=e^{\frac{z}{e}}$, the following:

$${\lambda}=1,\quad\frac{{\Upsilon}''(z)}{{\Upsilon}'(z)}=\frac{1}{e}$$
$$P'(z)=-\frac{f'(a)}{e}P(z)\sum_{t=0}^{\infty}{P(z+t)}$$

The parabolic_tetration.py implementation extracts $f'(a)$ numerically from the asymptotics of $P(z)$ for large $z$ that is initially constructed from the known orbit. Since $f'(a)P(z)=f'(z)$ with $f'(a)$, $P(z)$ and thus $f'(z)$ known (for discrete $z$), the following:

$$f''(z)=-\frac{1}{e}f'(z)\sum_{t=0}^{\infty}{f'(z+t)}$$
$$f^{(k+1)}(z)=-\frac{1}{e}\sum_{j=0}^{k-1}{\binom{k-1}{j}f^{(k-j)}(z)\sum_{t=0}^{\infty}{f^{(j+1)}(z+t)}}$$

Inductively, arbitrarily high order derivatives are expressible in terms of a convolution over shifted sums of lower order ones. Thus it is completely feasible that we precompute the orbit $f(z)$, the propagator $P(z)$ and from it $f'(a)$, $f'(z)$, and at last, any $f^{(k)}(z)$ via Leibniz binomials so that we can locally reconstruct both $f$ and $P$ around $z$ and meaningfully extend their domains to ℂ.
