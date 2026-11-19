import math
import warnings
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

DEFAULT_EXOG_COLS = ["SPY_r_oc", "SPY_r_vol", "SPY_r_overnight", "VIX_log"]


def _expected_abs_z(dist: str, nu: float = None) -> float:
    """Calcula E[|z|] para z ~ D(0, 1) estandarizado."""
    if dist == "normal":
        return math.sqrt(2.0 / math.pi)
    elif dist == "t":
        return (
            2.0
            * math.sqrt(nu - 2.0)
            * math.exp(gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0))
        ) / ((nu - 1.0) * math.sqrt(math.pi))
    elif dist == "ged":
        lam = math.sqrt((2.0 ** (-2.0 / nu)) * math.exp(gammaln(1.0 / nu) - gammaln(3.0 / nu)))
        return lam * (2.0 ** (1.0 / nu)) * math.exp(gammaln(2.0 / nu) - gammaln(1.0 / nu))
    raise ValueError(f"Distribución no soportada: {dist}")


def _filter_egarch_x_recursion(
    params: np.ndarray,
    r: np.ndarray,
    X_s: np.ndarray,
    p: int,
    o: int,
    q: int,
    dist: str,
    init_var: float,
):
    """
    Recursión causal de log-varianza condicional:
        ln(sigma_t^2) = omega + sum_i alpha_i*(|z_{t-i}| - E|z|)
                        + sum_j gamma_j*z_{t-j} + sum_k beta_k*ln(sigma_{t-k}^2)
                        + delta' X_{t-1}
    """
    T = len(r)
    k_exog = X_s.shape[1]

    mu = float(params[0])
    omega = float(params[1])
    alphas = [float(v) for v in params[2 : 2 + p]]
    gammas = [float(v) for v in params[2 + p : 2 + p + o]]
    betas = [float(v) for v in params[2 + p + o : 2 + p + o + q]]
    deltas = np.asarray(params[2 + p + o + q : 2 + p + o + q + k_exog], dtype=np.float64)
    nu = float(params[-1]) if dist in ("t", "ged") else None

    e_abs_z = _expected_abs_z(dist, nu)
    eps = r - mu
    init_log_var = math.log(max(init_var, 1e-4))

    base_h = np.empty(T, dtype=np.float64)
    base_h[0] = init_log_var
    if T > 1:
        base_h[1:] = omega + X_s[:-1] @ deltas

    log_sigma2 = [init_log_var] * T
    z = [0.0] * T
    z[0] = float(eps[0]) / math.sqrt(max(init_var, 1e-4))

    eps_list = eps.tolist()
    base_list = base_h.tolist()
    exp = math.exp

    for t in range(1, T):
        h_t = base_list[t]
        for i in range(p):
            idx = t - 1 - i
            if idx >= 0:
                h_t += alphas[i] * (abs(z[idx]) - e_abs_z)
        for j in range(o):
            idx = t - 1 - j
            if idx >= 0:
                h_t += gammas[j] * z[idx]
        for k in range(q):
            idx = t - 1 - k
            ht_k = log_sigma2[idx] if idx >= 0 else init_log_var
            h_t += betas[k] * ht_k

        if h_t < -15.0:
            h_t = -15.0
        elif h_t > 20.0:
            h_t = 20.0

        log_sigma2[t] = h_t
        z[t] = eps_list[t] * exp(-0.5 * h_t)

    return np.asarray(log_sigma2, dtype=np.float64), np.asarray(z, dtype=np.float64), nu


class EGARCHXModel:
    """
    Modelo econométrico EGARCH(p, o, q)-X estimado por Máxima Verosimilitud (MLE)
    con covariables exógenas rezagadas X_{t-1} directamente en la ecuación de log-varianza.
    """

    def __init__(self, p: int = 1, o: int = 1, q: int = 3, dist: str = "normal"):
        if dist not in ("normal", "ged", "t"):
            raise ValueError(f"dist debe ser 'normal', 'ged' o 't'. Recibido: {dist}")
        self.p = int(p)
        self.o = int(o)
        self.q = int(q)
        self.dist = dist

        self.scaler = RobustScaler()
        self.params_ = None
        self.init_var_ = None
        self.converged_ = False
        self.loglik_ = np.nan
        self.aic_ = np.nan
        self.bic_ = np.nan

    def _warm_start_univariate(self, r_train: np.ndarray, k_exog: int) -> list:
        """Obtiene punto inicial robusto desde el MLE univariado de arch_model."""
        default_mu = float(np.mean(r_train))
        default_omega = float(0.05 * math.log(max(np.var(r_train), 1e-4)))
        x0 = (
            [default_mu, default_omega]
            + [0.15 / self.p] * self.p
            + [-0.05] * self.o
            + [0.85 / self.q] * self.q
            + [0.0] * k_exog
        )
        if self.dist == "t":
            x0.append(5.0)
        elif self.dist == "ged":
            x0.append(1.2)

        try:
            am = arch_model(
                r_train,
                mean="Constant",
                vol="EGARCH",
                p=self.p,
                o=self.o,
                q=self.q,
                dist=self.dist,
            )
            res = am.fit(disp="off", show_warning=False)
            p_map = res.params.to_dict()

            mu_0 = float(np.clip(p_map.get("mu", default_mu), -4.5, 4.5))
            omega_0 = float(np.clip(p_map.get("omega", default_omega), -4.5, 4.5))
            alphas_0 = [float(np.clip(p_map.get(f"alpha[{i+1}]", 0.15 / self.p), -1.8, 1.8)) for i in range(self.p)]
            gammas_0 = [float(np.clip(p_map.get(f"gamma[{j+1}]", -0.05), -1.8, 1.8)) for j in range(self.o)]
            betas_0 = [float(np.clip(p_map.get(f"beta[{k+1}]", 0.85 / self.q), -0.98, 0.98)) for k in range(self.q)]

            warm = [mu_0, omega_0] + alphas_0 + gammas_0 + betas_0 + [0.0] * k_exog
            if self.dist == "t":
                warm.append(float(np.clip(p_map.get("nu", 5.0), 2.15, 45.0)))
            elif self.dist == "ged":
                warm.append(float(np.clip(p_map.get("nu", 1.2), 0.25, 9.5)))
            return warm
        except Exception:
            return x0

    def _neg_loglik(self, params: np.ndarray, r: np.ndarray, X_s: np.ndarray) -> float:
        log_sigma2, z, nu = _filter_egarch_x_recursion(
            params, r, X_s, self.p, self.o, self.q, self.dist, self.init_var_
        )
        if self.dist == "normal":
            ll = -0.5 * (math.log(2.0 * math.pi) + log_sigma2 + z * z)
        elif self.dist == "t":
            if nu <= 2.05:
                return 1e10
            c = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * math.log(math.pi * (nu - 2.0))
            ll = c - 0.5 * log_sigma2 - 0.5 * (nu + 1.0) * np.log1p((z * z) / (nu - 2.0))
        else:
            if nu <= 0.15:
                return 1e10
            lam = math.sqrt((2.0 ** (-2.0 / nu)) * math.exp(gammaln(1.0 / nu) - gammaln(3.0 / nu)))
            c = math.log(nu / lam) - (1.0 + 1.0 / nu) * math.log(2.0) - gammaln(1.0 / nu)
            ll = c - 0.5 * log_sigma2 - 0.5 * (np.abs(z / lam) ** nu)

        nll = -float(np.sum(ll[1:]))
        return nll if math.isfinite(nll) else 1e10

    def fit(self, r_train: np.ndarray, X_exog_train: np.ndarray):
        """
        Ajusta RobustScaler y parámetros MLE exclusivamente con datos in-sample.
        """
        r_tr = np.asarray(r_train, dtype=np.float64)
        X_tr = np.asarray(X_exog_train, dtype=np.float64)
        X_tr_s = self.scaler.fit_transform(X_tr)

        self.init_var_ = max(float(np.var(r_tr - np.mean(r_tr))), 1e-4)
        k_exog = X_tr_s.shape[1]

        x0 = np.asarray(self._warm_start_univariate(r_tr, k_exog), dtype=np.float64)
        bounds = (
            [(-5.0, 5.0), (-5.0, 5.0)]
            + [(-2.0, 2.0)] * self.p
            + [(-2.0, 2.0)] * self.o
            + [(-0.999, 0.999)] * self.q
            + [(-2.0, 2.0)] * k_exog
        )
        if self.dist == "t":
            bounds.append((2.1, 50.0))
        elif self.dist == "ged":
            bounds.append((0.2, 10.0))

        opt = minimize(
            self._neg_loglik,
            x0,
            args=(r_tr, X_tr_s),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 400, "ftol": 1e-9},
        )

        self.params_ = opt.x
        self.converged_ = bool(opt.success and math.isfinite(opt.fun))
        self.loglik_ = -float(opt.fun)
        n_obs = max(len(r_tr) - 1, 1)
        n_params = len(self.params_)
        self.aic_ = -2.0 * self.loglik_ + 2.0 * n_params
        self.bic_ = -2.0 * self.loglik_ + math.log(n_obs) * n_params
        return self

    def forecast_oos_1step(
        self,
        r_history_plus_oos: np.ndarray,
        X_history_plus_oos: np.ndarray,
        oos_start_idx: int,
    ) -> np.ndarray:
        """
        Genera pronósticos causales a 1 paso adelante (hat_sigma^2_{t+1 | t})
        para cada índice t en [oos_start_idx, len(r_history_plus_oos) - 1],
        usando estrictamente información observada hasta el cierre del día t.
        """
        if self.params_ is None:
            raise RuntimeError("El modelo debe entrenarse con .fit() antes de pronosticar.")

        r_all = np.asarray(r_history_plus_oos, dtype=np.float64)
        X_all = np.asarray(X_history_plus_oos, dtype=np.float64)
        X_all_s = self.scaler.transform(X_all)

        T = len(r_all)
        log_sigma2, z, nu = _filter_egarch_x_recursion(
            self.params_, r_all, X_all_s, self.p, self.o, self.q, self.dist, self.init_var_
        )

        # log_sigma2[s] contiene ln(sigma^2_{s | s-1}).
        # Por tanto, para t in [oos_start_idx, T-2], el pronóstico t+1|t está en log_sigma2[t+1].
        h_oos = np.empty(T - oos_start_idx, dtype=np.float64)
        if T - 1 > oos_start_idx:
            h_oos[:-1] = log_sigma2[oos_start_idx + 1 : T]

        # Último día del bloque (t = T - 1): un paso recursivo extra usando info hasta T - 1
        k_exog = X_all_s.shape[1]
        omega = float(self.params_[1])
        alphas = [float(v) for v in self.params_[2 : 2 + self.p]]
        gammas = [float(v) for v in self.params_[2 + self.p : 2 + self.p + self.o]]
        betas = [float(v) for v in self.params_[2 + self.p + self.o : 2 + self.p + self.o + self.q]]
        deltas = np.asarray(self.params_[2 + self.p + self.o + self.q : 2 + self.p + self.o + self.q + k_exog])
        e_abs_z = _expected_abs_z(self.dist, nu)

        h_last = omega + float(X_all_s[T - 1] @ deltas)
        for i in range(self.p):
            idx = T - 1 - i
            if idx >= 0:
                h_last += alphas[i] * (abs(float(z[idx])) - e_abs_z)
        for j in range(self.o):
            idx = T - 1 - j
            if idx >= 0:
                h_last += gammas[j] * float(z[idx])
        for k in range(self.q):
            idx = T - 1 - k
            if idx >= 0:
                h_last += betas[k] * float(log_sigma2[idx])

        h_oos[-1] = float(np.clip(h_last, -15.0, 20.0))
        return np.exp(h_oos)

    def get_params_dict(self, exog_cols: list) -> dict:
        """Devuelve un diccionario nombrado de todos los parámetros estimados."""
        p, o, q = self.p, self.o, self.q
        k_exog = len(exog_cols)
        d = {
            "mu": float(self.params_[0]),
            "omega": float(self.params_[1]),
        }
        for i in range(p):
            d[f"alpha_{i+1}"] = float(self.params_[2 + i])
        for j in range(o):
            d[f"gamma_{j+1}"] = float(self.params_[2 + p + j])
        for k in range(q):
            d[f"beta_{k+1}"] = float(self.params_[2 + p + o + k])
        for m, col in enumerate(exog_cols):
            d[f"delta_{col}"] = float(self.params_[2 + p + o + q + m])
        d["nu"] = float(self.params_[-1]) if self.dist in ("t", "ged") else np.nan
        return d