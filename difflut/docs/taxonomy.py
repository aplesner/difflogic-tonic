from tikzpy import TikzPicture, Rectangle, Point

tikz = TikzPicture(center=True)

# --- Paper size scaling ---
scale_factor = 1.2

# Parameters (scaled by paper size)
box_w, box_h = 3 * scale_factor, 1 * scale_factor
x_gap, y_gap = 1 * scale_factor, 1.5 * scale_factor

# User-defined connections between Weight Parametization and Forward Pass Function
param_to_fwd_arrows = [
    (0, 0),  # 2^{2^k} -> DiffLogicNet
    (1, 1),  # 2^k -> DWN
    (1, 2),  # 2^k -> Probabilistic
    (2, 3),  # 2^{βF} -> LogicNets
    (3, 4),  # \binom{F+D}{D} -> PolyLUT
    (4, 5),  # N -> NeuralLUT
    (5, 6),  # 2^k -> LUTNet/XNOR
    (5, 7),  # 2^k -> NullaNet Boolean
]

# Layer titles and content - UPDATED TO MATCH PAPER NOTATION
layer_defs = [
    ("Mappings", ["Random", "Learnable", "Adaptive"]),
    ("Weight\\ Parametization", [
        r"$\omega \in \mathbb{R}^{2^{2^k}}$",  # DiffLogicNet
        r"$\omega \in \mathbb{R}^{2^k}$",      # DWN
        r"$\omega \in \mathbb{R}^{2^{\beta F}}$",  # LogicNets
        r"$\omega \in \mathbb{R}^{\binom{F+D}{D}}$",  # PolyLUT
        r"$\omega \in \mathbb{R}^{N}$",        # NeuralLUT
        r"$\omega \in \mathbb{R}^{2^k}$",      # LUTNet/XNOR, NullaNet
    ]),
    ("Forward Pass Function", [
        # DiffLogicNet - matches your paper's probabilistic formulation
        r"DiffLogicNet:\\ $y = \sum_{i=0}^{15} \frac{e^{w_i}}{\sum_j e^{w_j}} f_i(x_1,x_2)$",
        # DWN - discrete selection
        r"DWN:\\ $y = \omega_{\iota(x)}$",
        # Probabilistic - multilinear extension from your paper
        r"Probabilistic:\\ $y = \sum_{a\in\{0,1\}^k} \omega_{\iota(a)} \prod_{j=1}^k x_j^{a_j}(1-x_j)^{1-a_j}$",
        # LogicNets - from survey
        r"LogicNets:\\ $y = \sigma\left(\sum_{i=0}^{F-1} w_ix_i + b\right)$",
        # PolyLUT - from survey
        r"PolyLUT:\\ $y = \sigma\left(\sum_{i=0}^{M-1} w_im_i(x) + b\right)$",
        # NeuralLUT - from survey
        r"NeuralLUT:\\ $y = F_L \circ \phi \circ F_{L-1} \circ \cdots \circ F_1(x)$",
        # LUTNet/XNOR - from survey
        r"LUTNet/XNOR:\\ $y = \sigma\left(\sum_{i=1}^N \text{XNOR}(x_i,w_i)\right)$",
        # NullaNet Boolean - from survey
        r"NullaNet Boolean:\\ $y = \mathbf{1}_{\{\sum w_ix_i \geq b\}}$",
    ]),
    ("Gradients w.r.t. Weights", [
        # DiffLogicNet - softmax gradient
        r"DiffLogicNet:\\ $\frac{\partial y}{\partial w_i} = \frac{1}{\tau} p_i(f_i - \sum_j p_j f_j)$",
        # DWN - indicator gradient
        r"DWN:\\ $\frac{\partial y}{\partial \omega_i} = \mathbf{1}_{\{i = \iota(x)\}}$",
        # Probabilistic - Bernoulli basis
        r"Probabilistic:\\ $\frac{\partial y}{\partial \omega_i} = \prod_{j=1}^k x_j^{a_j^{(i)}}(1-x_j)^{1-a_j^{(i)}}$",
        # LogicNets - linear gradient
        r"LogicNets:\\ $\frac{\partial y}{\partial w_i} = x_i \cdot \sigma'$",
        # PolyLUT - polynomial gradient
        r"PolyLUT:\\ $\frac{\partial y}{\partial w_i} = m_i(x) \cdot \sigma'$",
        # NeuralLUT - backpropagation
        r"NeuralLUT:\\ $\frac{\partial y}{\partial w} = \text{Backprop through MLP}$",
        # LUTNet/XNOR - straight-through
        r"LUTNet/XNOR:\\ $\frac{\partial y}{\partial w_i} = x_i \cdot \sigma'$",
        # NullaNet Boolean - threshold gradient
        r"NullaNet Boolean:\\ $\frac{\partial y}{\partial w_i} = x_i \cdot \delta_{\text{threshold}}$",
    ]),
    ("Gradients w.r.t. Inputs", [
        # DiffLogicNet - mixture gradient
        r"DiffLogicNet:\\ $\frac{\partial y}{\partial x_j} = \sum_i p_i \frac{\partial f_i}{\partial x_j}$",
        # DWN - EFD method from survey
        r"DWN:\\ $\frac{\partial y}{\partial x_j} = \sum_{k\in\{0,1\}^n} (-1)^{1-k_j}A(U,k)\cdot H(k,x,j) + 1$",
        # Probabilistic - product rule
        r"Probabilistic:\\ $\frac{\partial y}{\partial x_j} = \sum_a \omega_{\iota(a)} \frac{\partial}{\partial x_j}\prod_{l=1}^k x_l^{a_l}(1-x_l)^{1-a_l}$",
        # LogicNets - linear gradient
        r"LogicNets:\\ $\frac{\partial y}{\partial x_j} = w_j \cdot \sigma'$",
        # PolyLUT - polynomial derivative
        r"PolyLUT:\\ $\frac{\partial y}{\partial x_j} = \sum_i w_i \frac{\partial m_i}{\partial x_j} \cdot \sigma'$",
        # NeuralLUT - backpropagation
        r"NeuralLUT:\\ $\frac{\partial y}{\partial x} = \text{Backprop through MLP}$",
        # LUTNet/XNOR - straight-through
        r"LUTNet/XNOR:\\ $\frac{\partial y}{\partial x_j} = w_j \cdot \sigma'$",
        # NullaNet Boolean - threshold gradient
        r"NullaNet Boolean:\\ $\frac{\partial y}{\partial x_j} = w_j \cdot \delta_{\text{threshold}}$",
    ]),
    ("Regularization", ["Spectral Norm", "L-Norm", "Ensemble"]),
    ("Initialization", ["Residual", "Gaussian", "Kaiming"]),
]

# Custom connections from Forward Pass to Gradients w.r.t. Weights
fwd_to_weightgrad_arrows = [
    (i, i) for i in range(len(layer_defs[2][1]))
]
fwd_to_weightgrad_arrows.extend([(1, 2),(2,1)])  # DWN -> Probabilistic

# Custom connections from Gradients w.r.t. Weights to Gradients w.r.t. Inputs
weightgrad_to_inputgrad_arrows = [
    (i, i) for i in range(len(layer_defs[3][1]))
]
weightgrad_to_inputgrad_arrows.extend([(1, 2),(2,1)])  # DWN -> Probabilistic


# Colors for layer backgrounds
layer_colors = ["red!5", "orange!5", "blue!5", "green!5", "purple!5", "gray!5", "yellow!5"]

# Compute max width for all layers
max_n_boxes = max(len(boxes) for _, boxes in layer_defs)
total_w = max_n_boxes * box_w + (max_n_boxes - 1) * x_gap
layerbox_h = box_h + 1

# Helper: make a box with text
def make_box(center, text):
    rect = tikz.rectangle_from_center(
        center,
        width=box_w,
        height=box_h,
        options="fill=white, draw=black, rounded corners=6pt, ultra thick"
    )
    # If text contains $ (LaTeX math), use math mode and smaller font
    if "$" in text:
        tikz.node(rect.center, options="align=center, font=\\scriptsize", text=text)
    else:
        tikz.node(rect.center, options="align=center, font=\\scriptsize", text=text)
    return rect

# Helper: add background layer (all same size, left-aligned for title)
def add_layer_bg(y, color):
    # Shift center right to leave space for title
    x_shift = 1.5 * box_w
    center = Point(x_shift / 2, y)
    rect = tikz.rectangle_from_center(
        center,
        width=total_w + x_shift,
        height=layerbox_h,
        options=f"fill={color}, opacity=0.5, draw=none, rounded corners=10pt"
    )
    return rect

# Helper: add layer title
def add_layer_title(y, title):
    # Place title to the left of the layer box
    x = -0.5 * total_w
    y = y + 0.5 * box_h
    tikz.node(Point(x, y), options="anchor=west, align=left", text=title)

# --- Draw all layers ---
layer_boxes = []
layer_y = 0
layer_gap = y_gap + box_h
for i, (title, labels) in enumerate(layer_defs):
    add_layer_bg(layer_y, layer_colors[i % len(layer_colors)])
    boxes = []
    n = len(labels)
    # Shift only the 'Gradients w.r.t. Weights' layer (index 3) to the right
    for j, lbl in enumerate(labels):
        # Center boxes in the available width
        x = (j - (n-1)/2) * (box_w + x_gap) + 1.5 * box_w / 2
        boxes.append(make_box(Point(x, layer_y), lbl))
    layer_boxes.append(boxes)
    add_layer_title(layer_y, title)
    layer_y -= layer_gap

# Draw vertical arrows
def connect_boxes(top, bottom):
    tikz.line(top.south, bottom.north, options="ultra thick, ->, >=stealth")

# Connect Weight Parametization to Forward Pass Function
param_boxes = layer_boxes[1]
fwd_boxes = layer_boxes[2]
for p_idx, f_idx in param_to_fwd_arrows:
    connect_boxes(param_boxes[p_idx], fwd_boxes[f_idx])

# Custom connections: Forward Pass -> Gradients w.r.t. Weights
weight_grad_boxes = layer_boxes[3]
for fwd_idx, grad_idx in fwd_to_weightgrad_arrows:
    if fwd_idx < len(fwd_boxes) and grad_idx < len(weight_grad_boxes):
        tikz.line(fwd_boxes[fwd_idx].south, weight_grad_boxes[grad_idx].north, 
                 options="ultra thick, ->, >=stealth")

# Custom connections: Gradients w.r.t. Weights -> Gradients w.r.t. Inputs
input_grad_boxes = layer_boxes[4]
for grad_idx, input_idx in weightgrad_to_inputgrad_arrows:
    if grad_idx < len(weight_grad_boxes) and input_idx < len(input_grad_boxes):
        tikz.line(weight_grad_boxes[grad_idx].south, input_grad_boxes[input_idx].north, 
                 options="ultra thick, ->, >=stealth")

# Connect all other layers vertically
for i in range(len(layer_boxes) - 1):
    # Skip already connected layers
    if i in [1, 2, 3]:  # Skip parametization->forward, fwd->weightgrad, weightgrad->inputgrad
        continue
    for box1 in layer_boxes[i]:
        for box2 in layer_boxes[i+1]:
            connect_boxes(box1, box2)

with open("taxonomy_tikz.txt", "w") as f:
    f.write(tikz.code())
# tikz.show()