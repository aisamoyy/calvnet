# calvnet
# CalVNet: Learning Variational Optimal Control via Deep Networks

CalVNet is a deep learning framework for solving dynamic optimization problems by integrating **calculus of variations** directly into neural network training. Rather than relying on labeled control data, CalVNet learns optimal trajectories by minimizing residuals from the necessary conditions of optimality (e.g., Pontryagin's Maximum Principle). This approach enables data-efficient learning of control strategies, geodesics, and transport dynamics under complex constraints.

---

## 🔍 Key Features

- ✅ No supervision from state/control labels required
- ✅ Incorporates state/constraints conditions by design
- ✅ Solves optimal control, filtering, and geodesic problems
- ✅ Compatible with variational formulations 

---

## 🧠 How It Works

CalVNet learns optimal solutions that adhere to the **necessary conditions** of the variational problem (e.g., PMP). The neural network outputs state, control, and costate variables simultaneously, and the training loss penalizes violations of the governing differential equations and boundary constraints.

---

## 📦 Installation

```bash
git clone https://github.com/aisamoyy/calvnet.git
cd calvnet
pip install -r requirements.txt
