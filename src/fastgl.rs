use crate::data;
use std::f64::consts::PI;

fn try_usize_into_f64(x: usize) -> Result<f64, ()> {
    let result = x as f64;
    if result as usize != x {
        return Err(());
    }
    Ok(result)
}

/// Returns the k:th zero of BesselJ[0, x]
fn besselj0_zero(k: usize) -> f64 {
    if k > 20 {
        let z: f64 = PI * (k as f64 - 0.25);
        let r = 1.0 / z;
        let r2 = r * r;
        z + r
            * (0.125
                + r2 * (-0.807291666666666666666666666667e-1
                    + r2 * (0.246028645833333333333333333333
                        + r2 * (-1.82443876720610119047619047619
                            + r2 * (25.3364147973439050099206349206
                                + r2 * (-567.644412135183381139802038240
                                    + r2 * (18690.4765282320653831636345064
                                        + r2 * (-8.49353580299148769921876983660e5
                                            + 5.09225462402226769498681286758e7 * r2))))))))
    } else {
        data::JZ[k - 1]
    }
}

/// Returns BesselJ[1, BesselZero[0, k]]^2
fn besselj1_squared(k: usize) -> f64 {
    if k > 21 {
        let x: f64 = 1.0 / (k as f64 - 0.25);
        let x2 = x * x;
        x * (0.202642367284675542887758926420
            + x2 * x2
                * (-0.303380429711290253026202643516e-3
                    + x2 * (0.198924364245969295201137972743e-3
                        + x2 * (-0.228969902772111653038747229723e-3
                            + x2 * (0.433710719130746277915572905025e-3
                                + x2 * (-0.123632349727175414724737657367e-2
                                    + x2 * (0.496101423268883102872271417616e-2
                                        + x2 * (-0.266837393702323757700998557826e-1
                                            + 0.185395398206345628711318848386 * x2))))))))
    } else {
        data::J1[k - 1]
    }
}

#[derive(Debug, Clone, PartialEq)]
struct QuadPair {
    theta: f64,
    weight: f64,
}

use core::num::NonZeroUsize;
impl QuadPair {
    pub fn new(n: NonZeroUsize, k: NonZeroUsize) -> Self {
        assert!(n >= k);
        let n = n.get();
        let k = k.get();
        if n < 101 {
            Self::gl_pair_tabulated(n, k - 1)
        } else {
            if 2* k - 1 > n {
                let mut p = Self::gl_pair_s(n, n - k + 1);
                p.theta = PI - p.theta;
                p
            } else {
                Self::gl_pair_s(n, k)
            }
        }
    }

    fn x(&self) -> f64 {
        self.theta.cos()
    }

    fn gl_pair_s(n: usize, k: usize) -> Self {
        let w: f64 = 1.0 / (n as f64 + 0.5);
        let nu = besselj0_zero(k);
        let theta = w * nu;
        let x = theta * theta;

        let b = besselj1_squared(k);

        let sf1t = (((((-1.29052996274280508473467968379e-12 * x
            + 2.40724685864330121825976175184e-10)
            * x
            - 3.13148654635992041468855740012e-8)
            * x
            + 0.275573168962061235623801563453e-5)
            * x
            - 0.148809523713909147898955880165e-3)
            * x
            + 0.416666666665193394525296923981e-2)
            * x
            - 0.416666666666662959639712457549e-1;
        let sf2t = (((((2.20639421781871003734786884322e-9 * x
            - 7.53036771373769326811030753538e-8)
            * x
            + 0.161969259453836261731700382098e-5)
            * x
            - 0.253300326008232025914059965302e-4)
            * x
            + 0.282116886057560434805998583817e-3)
            * x
            - 0.209022248387852902722635654229e-2)
            * x
            + 0.815972221772932265640401128517e-2;
        let sf3t = (((((-2.97058225375526229899781956673e-8 * x
            + 5.55845330223796209655886325712e-7)
            * x
            - 0.567797841356833081642185432056e-5)
            * x
            + 0.418498100329504574443885193835e-4)
            * x
            - 0.251395293283965914823026348764e-3)
            * x
            + 0.128654198542845137196151147483e-2)
            * x
            - 0.416012165620204364833694266818e-2;

        let wsf1t = ((((((((-2.20902861044616638398573427475e-14 * x
            + 2.30365726860377376873232578871e-12)
            * x
            - 1.75257700735423807659851042318e-10)
            * x
            + 1.03756066927916795821098009353e-8)
            * x
            - 4.63968647553221331251529631098e-7)
            * x
            + 0.149644593625028648361395938176e-4)
            * x
            - 0.326278659594412170300449074873e-3)
            * x
            + 0.436507936507598105249726413120e-2)
            * x
            - 0.305555555555553028279487898503e-1)
            * x
            + 0.833333333333333302184063103900e-1;
        let wsf2t = (((((((3.63117412152654783455929483029e-12 * x
            + 7.67643545069893130779501844323e-11)
            * x
            - 7.12912857233642220650643150625e-9)
            * x
            + 2.11483880685947151466370130277e-7)
            * x
            - 0.381817918680045468483009307090e-5)
            * x
            + 0.465969530694968391417927388162e-4)
            * x
            - 0.407297185611335764191683161117e-3)
            * x
            + 0.268959435694729660779984493795e-2)
            * x
            - 0.111111111111214923138249347172e-1;
        let wsf3t = (((((((2.01826791256703301806643264922e-9 * x
            - 4.38647122520206649251063212545e-8)
            * x
            + 5.08898347288671653137451093208e-7)
            * x
            - 0.397933316519135275712977531366e-5)
            * x
            + 0.200559326396458326778521795392e-4)
            * x
            - 0.422888059282921161626339411388e-4)
            * x
            - 0.105646050254076140548678457002e-3)
            * x
            - 0.947969308958577323145923317955e-4)
            * x
            + 0.656966489926484797412985260842e-2;

        let nu_o_sin = nu / theta.sin();
        let b_nu_o_sin = b * nu_o_sin;
        let w_inv_sinc = w * w * nu_o_sin;
        let wis2 = w_inv_sinc * w_inv_sinc;

        let theta = w * (nu + theta + w_inv_sinc * (sf1t + wis2 * (sf2t + wis2 * sf3t)));
        let deno = b_nu_o_sin + b_nu_o_sin * wis2 * (wsf1t + wis2 * (wsf2t + wis2 * wsf3t));
        let weight = 2.0 * w / deno;

        Self { theta, weight }
    }

    fn gl_pair_tabulated(l: usize, k: usize) -> Self {
        use data::{CL, EVEN_THETA_ZEROS, EVEN_WEIGHTS, ODD_THETA_ZEROS, ODD_WEIGHTS};
        let (theta, weight) = if l % 2 == 1 {
            // originally l & 1
            let l2 = (l - 1) / 2;
            if k == l2 {
                (PI / 2.0, 2.0 / (CL[l] * CL[l]))
            } else if k < l2 {
                (
                    ODD_THETA_ZEROS[l2 - 1][l2 - k - 1],
                    ODD_WEIGHTS[l2 - 1][l2 - k - 1],
                )
            } else {
                (
                    PI - ODD_THETA_ZEROS[l2 - 1][k - l2 - 1],
                    ODD_WEIGHTS[l2 - 1][k - l2 - 1],
                )
            }
        } else {
            let l2 = l / 2;
            if k < l2 {
                (
                    EVEN_THETA_ZEROS[l2 - 1][l2 - k - 1],
                    EVEN_WEIGHTS[l2 - 1][l2 - k - 1],
                )
            } else {
                (
                    PI - EVEN_THETA_ZEROS[l2 - 1][k - l2],
                    EVEN_WEIGHTS[l2 - 1][k - l2],
                )
            }
        };
        Self { theta, weight }
    }
}
