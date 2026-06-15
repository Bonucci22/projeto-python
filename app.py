import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sympy as sp
from flask import Flask, render_template, request

from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)

app = Flask(__name__)
os.makedirs(os.path.join('static', 'images'), exist_ok=True)

def criar_funcao_e_derivada(expressao):
    x = sp.symbols("x")
    funcoes_permitidas = {
        "x": x, "log": sp.log, "ln": sp.log,
        "log10": lambda argumento: sp.log(argumento, 10),
        "exp": sp.exp, "sin": sp.sin, "cos": sp.cos,
        "tan": sp.tan, "sqrt": sp.sqrt, "abs": sp.Abs,
        "pi": sp.pi, "e": sp.E
    }
    transformacoes = standard_transformations + (implicit_multiplication_application, convert_xor)
    funcao_simbolica = parse_expr(expressao, local_dict=funcoes_permitidas, transformations=transformacoes)
    derivada_simbolica = sp.diff(funcao_simbolica, x)
    funcao_numerica = sp.lambdify(x, funcao_simbolica, modules=["numpy"])
    derivada_numerica = sp.lambdify(x, derivada_simbolica, modules=["numpy"])
    return funcao_simbolica, derivada_simbolica, funcao_numerica, derivada_numerica

def bisseccao(func, a, b, tol=1e-8, max_iter=1000):
    fa, fb = float(func(a)), float(func(b))
    if np.abs(fa) < tol: return a, 0
    if np.abs(fb) < tol: return b, 0
    if fa * fb > 0: raise ValueError("f(a) e f(b) devem ter sinais opostos no intervalo escolhido.")
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = float(func(c))
        if np.abs(fc) < tol or (b - a) / 2 < tol: return c, i
        if fa * fc < 0: b, fb = c, fc
        else: a, fa = c, fc
    return c, max_iter

def gerar_grafico(func, derivada, a, b, raiz):
    plt.clf()
    y_raiz = float(func(raiz))
    m_tang = float(derivada(raiz))
    
    largura = max(b - a, 1.0)
    x_inicio = a - largura * 0.1
    x_fim = b + largura * 0.1
    x = np.linspace(x_inicio, x_fim, 1000)
    
    with np.errstate(all="ignore"):
        y = np.asarray(func(x), dtype=float)
        y = np.where(np.isfinite(y), y, np.nan)

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')

    ax.plot(x, y, color='#38bdf8', lw=3, label="f(x)")
    
    y_tan = m_tang * (x - raiz) + y_raiz
    ax.plot(x, y_tan, "--", color='#4ade80', lw=2, label="Reta Tangente")

    if abs(m_tang) > 1e-10:
        m_norm = -1/m_tang
        y_norm = m_norm * (x - raiz) + y_raiz
        ax.plot(x, y_norm, "-.", color='#f43f5e', lw=2, label="Reta Normal")
    else:
        ax.axvline(raiz, color='#f43f5e', ls='-.', lw=2, label="Reta Normal")

    ax.axhline(0, color='#475569', lw=1, alpha=0.5)
    ax.axvline(0, color='#475569', lw=1, alpha=0.5)
    ax.plot(raiz, y_raiz, "o", color='#facc15', ms=10, label=f"Raiz: {raiz:.4f}")
    
    y_lims = y[(x >= a) & (x <= b) & (~np.isnan(y))]
    if len(y_lims) > 0:
        folga = max(abs(max(y_lims) - min(y_lims)) * 0.2, 1.0)
        ax.set_ylim(min(y_lims) - folga, max(y_lims) + folga)
    else:
        ax.set_ylim(-5, 5)
        
    ax.set_xlim(x_inicio, x_fim)
    ax.set_aspect('equal', adjustable='datalim')
    
    ax.tick_params(colors='#94a3b8')
    ax.grid(True, ls=':', color='#334155', alpha=0.4)
    ax.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='white', loc='upper right')
    
    plt.tight_layout()
    caminho = os.path.join('static', 'images', 'grafico.png')
    plt.savefig(caminho, dpi=130, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    res = None
    erro = None
    if request.method == 'POST':
        try:
            exp = request.form.get('expressao')
            a = float(request.form.get('a_lim'))
            b = float(request.form.get('b_lim'))
            
            f_simb, d_simb, f_num, d_num = criar_funcao_e_derivada(exp)
            raiz, iters = bisseccao(f_num, a, b)
            
            y0 = float(f_num(raiz))
            m = float(d_num(raiz))
            
            # Formatação estrutural condicional para ocultar resíduos próximos de zero
            if abs(y0) < 1e-4:
                eq_t = f"y = {m:.4f}(x - {raiz:.4f})"
                if abs(m) < 1e-10:
                    eq_n = f"x = {raiz:.4f}"
                else:
                    eq_n = f"y = {-1/m:.4f}(x - {raiz:.4f})"
            else:
                sinal_y0 = "+" if y0 >= 0 else "-"
                eq_t = f"y = {m:.4f}(x - {raiz:.4f}) {sinal_y0} {abs(y0):.4f}"
                if abs(m) < 1e-10:
                    eq_n = f"x = {raiz:.4f}"
                else:
                    eq_n = f"y = {-1/m:.4f}(x - {raiz:.4f}) {sinal_y0} {abs(y0):.4f}"
            
            gerar_grafico(f_num, d_num, a, b, raiz)
            
            res = {
                "exp_original": exp,
                "f_formatada": sp.sstr(f_simb),
                "d_formatada": sp.sstr(d_simb),
                "raiz": f"{raiz:.6f}",
                "iters": iters,
                "acuracia": f"{y0:.2e}",
                "eq_tangente": eq_t,
                "eq_normal": eq_n,
                "grafico_url": "/static/images/grafico.png"
            }
        except (sp.sympify.SympifyError, TypeError, SyntaxError, NameError):
            # Intercepta erros estruturais da equação matemática digitada pelo usuário
            erro = "Erro de digitação, digite a função novamente."
        except Exception as e:
            # Mantém outros erros matemáticos operacionais (como f(a) e f(b) sem inversão de sinal)
            erro = str(e)
            
    return render_template('index.html', res=res, erro=erro)

if __name__ == '__main__':
    app.run(debug=True)