/* ── Architecture info panel ────────────────────────── */
const archInfo = {
  'embed-src': {
    title: 'Source Embedding',
    body:  'Converts integer token IDs into dense d_model dimensional vectors. Weights are shared with the target output projection.',
    spec:  'input_dim  : vocab_size\noutput_dim : 512 (d_model)\nscaling    : x * sqrt(d_model)\nsharing    : tied with output projection'
  },
  'embed-tgt': {
    title: 'Target Embedding',
    body:  'Maps target token IDs to dense representations. Shared with encoder embeddings in our implementation.',
    spec:  'input_dim  : vocab_size\noutput_dim : 512 (d_model)\nscaling    : x * sqrt(d_model)\nsharing    : tied with encoder embedding'
  },
  'pe': {
    title: 'Sinusoidal Positional Encoding',
    body:  'Fixed sinusoidal signal injected into token representations to encode absolute and relative sequence positions.',
    spec:  'PE(pos,2i)   = sin(pos/10000^(2i/d_model))\nPE(pos,2i+1) = cos(pos/10000^(2i/d_model))\nmax_len     : 5000\nparams      : 0 (fixed)'
  },
  'enc-layer': {
    title: 'Encoder Layer (x6)',
    body:  'Contains multi-head self-attention and position-wise feed-forward sub-layers with residual connections and POST-LN.',
    spec:  'sub-layer 1 : MultiHeadAttention(x,x,x)\nsub-layer 2 : FFN(x)\nwrap        : LayerNorm(x + Dropout(SubLayer(x)))\nnorm order  : POST-LN\nparams/layer: ~2.1M'
  },
  'dec-layer': {
    title: 'Decoder Layer (x6)',
    body:  'Includes masked self-attention, cross-attention over encoder output, and feed-forward sub-layers.',
    spec:  'sub-layer 1 : Masked MHA(x,x,x)\nsub-layer 2 : Cross MHA(x,enc,enc)\nsub-layer 3 : FFN(x)\nnorm order  : POST-LN\nparams/layer: ~3.1M'
  },
  'enc-out': {
    title: 'Encoder Output Stack',
    body:  'Final contextualized sequence representation fed into every decoder cross-attention sub-layer.',
    spec:  'shape : (batch, src_len, 512)\nused  : Key (K) and Value (V) in all 6 decoder layers'
  },
  'proj': {
    title: 'Linear Projection Layer',
    body:  'Transforms decoder hidden representation into output vocabulary logit values.',
    spec:  'in_dim  : 512 (d_model)\nout_dim : vocab_size\nbias    : False\nweight  : tied with embedding matrix'
  },
  'logits': {
    title: 'Output Logits',
    body:  'Unnormalized class scores over vocabulary size per generated sequence token.',
    spec:  'shape    : (batch, tgt_len, vocab_size)\ntraining : cross-entropy with label smoothing\ninference: greedy argmax or beam search'
  },
};

function showInfo(key) {
  const info = archInfo[key];
  if (!info) return;

  document.querySelectorAll('.arch-block').forEach(el => el.classList.remove('active'));
  const activeEl = document.getElementById('blk-' + key);
  if (activeEl) activeEl.classList.add('active');

  document.getElementById('arch-info-title').textContent = info.title;
  document.getElementById('arch-info-body').textContent  = info.body;
  document.getElementById('arch-info-spec').innerHTML    =
    info.spec.split('\n').map(line => {
      const parts = line.split(':');
      if (parts.length < 2) return line;
      return `<span class="key">${parts[0]}</span>:${parts.slice(1).join(':')}`;
    }).join('<br>');
}

/* ── Loss Chart Drawing Routine ─────────────────────── */
function drawLossChart() {
  const canvas = document.getElementById('loss-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W   = canvas.offsetWidth || 460;
  const H   = 200;

  canvas.width  = W * window.devicePixelRatio;
  canvas.height = H * window.devicePixelRatio;
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

  const trainData = [
    [100,5.38],[200,4.71],[300,4.68],[400,4.80],[500,4.45],
    [600,4.17],[700,4.00],[800,4.22],[900,3.94],[1000,3.65],
    [1500,3.14],[2000,2.98],[2500,2.89],[3000,2.94],[3500,2.55],
    [4000,2.88],[4500,2.48],[5000,2.60],[5500,2.25],[6000,2.39],
    [6500,2.28],[7000,2.17],[7500,2.24],[8000,2.18],[8500,2.07],
    [9000,2.41],[9500,2.10],[10000,2.14]
  ];
  const valData = [
    [500,4.36],[1000,3.72],[1500,3.41],[2000,3.30],[2500,3.13],
    [3000,3.01],[3500,2.96],[4000,2.87],[4500,2.83],[5000,2.80],
    [5500,2.76],[6000,2.73],[6500,2.69],[7000,2.68],[7500,2.67],
    [8000,2.68],[8500,2.65],[9000,2.63],[9500,2.64],[10000,2.64]
  ];

  const pad = { t:20, r:20, b:36, l:44 };
  const pw  = W - pad.l - pad.r;
  const ph  = H - pad.t - pad.b;
  const minY = 1.5, maxY = 5.8;
  const minX = 0, maxX = 10000;

  function tx(x) { return pad.l + (x - minX) / (maxX - minX) * pw; }
  function ty(y) { return pad.t + (1 - (y - minY) / (maxY - minY)) * ph; }

  // Background Grid
  ctx.strokeStyle = '#E2E8F0';
  ctx.lineWidth   = 1;
  [2,3,4,5].forEach(y => {
    ctx.beginPath();
    ctx.moveTo(pad.l, ty(y));
    ctx.lineTo(pad.l + pw, ty(y));
    ctx.stroke();
    ctx.fillStyle = '#64748B';
    ctx.font      = '10px IBM Plex Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(y.toFixed(0), pad.l - 8, ty(y) + 3);
  });

  // X axis labels
  [0,2000,4000,6000,8000,10000].forEach(x => {
    ctx.fillStyle = '#64748B';
    ctx.font      = '10px IBM Plex Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(x === 0 ? '0' : (x/1000)+'k', tx(x), H - 8);
  });

  function drawLine(data, color, width=2, dash=[]) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth   = width;
    ctx.setLineDash(dash);
    data.forEach(([x,y], i) => {
      i === 0 ? ctx.moveTo(tx(x), ty(y)) : ctx.lineTo(tx(x), ty(y));
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Draw curves
  drawLine(trainData, '#2563EB', 2.5);
  drawLine(valData,   '#059669', 2, [4,3]);

  // Legend
  ctx.fillStyle = '#2563EB'; ctx.fillRect(pad.l, 4, 16, 3);
  ctx.fillStyle = '#0F172A'; ctx.font = '10px IBM Plex Sans, sans-serif'; ctx.textAlign = 'left';
  ctx.fillText('Train Loss', pad.l + 22, 8);

  ctx.fillStyle = '#059669'; ctx.fillRect(pad.l + 100, 4, 16, 3);
  ctx.fillText('Val Loss', pad.l + 122, 8);
}

/* ── LR Schedule Chart ───────────────────────────── */
function drawLRChart() {
  const canvas = document.getElementById('lr-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W   = canvas.offsetWidth || 460;
  const H   = 130;

  canvas.width  = W * window.devicePixelRatio;
  canvas.height = H * window.devicePixelRatio;
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

  function lr(step, d=256, w=2000) {
    step = Math.max(step, 1);
    return Math.pow(d, -0.5) * Math.min(Math.pow(step, -0.5), step * Math.pow(w, -1.5));
  }

  const steps = [];
  for (let s = 1; s <= 10000; s += 100) steps.push(s);
  const lrs   = steps.map(s => lr(s));
  const maxLR = Math.max(...lrs);

  const pad = { t:12, r:20, b:28, l:52 };
  const pw  = W - pad.l - pad.r;
  const ph  = H - pad.t - pad.b;

  function tx(i) { return pad.l + (i / (steps.length - 1)) * pw; }
  function ty(v) { return pad.t + (1 - v / (maxLR * 1.1)) * ph; }

  ctx.strokeStyle = '#E2E8F0';
  ctx.lineWidth   = 1;
  [0, maxLR * 0.5, maxLR].forEach(v => {
    ctx.beginPath();
    ctx.moveTo(pad.l, ty(v));
    ctx.lineTo(pad.l + pw, ty(v));
    ctx.stroke();
    ctx.fillStyle = '#64748B';
    ctx.font      = '9px IBM Plex Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(v.toExponential(0), pad.l - 6, ty(v) + 3);
  });

  [0,2000,4000,6000,8000,10000].forEach(s => {
    const i = steps.findIndex(x => x >= s);
    if (i < 0) return;
    ctx.fillStyle = '#64748B';
    ctx.font      = '9px IBM Plex Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(s === 0 ? '0' : (s/1000)+'k', tx(i), H - 6);
  });

  // LR curve
  ctx.beginPath();
  ctx.strokeStyle = '#7C3AED';
  ctx.lineWidth   = 2.2;
  lrs.forEach((v, i) => i === 0 ? ctx.moveTo(tx(i), ty(v)) : ctx.lineTo(tx(i), ty(v)));
  ctx.stroke();

  // Warmup marker line
  const warmupIdx = steps.findIndex(s => s >= 2000);
  ctx.beginPath();
  ctx.strokeStyle = '#D97706';
  ctx.lineWidth   = 1;
  ctx.setLineDash([3,3]);
  ctx.moveTo(tx(warmupIdx), pad.t);
  ctx.lineTo(tx(warmupIdx), pad.t + ph);
  ctx.stroke();
  ctx.setLineDash([]);
  
  ctx.fillStyle = '#D97706';
  ctx.font      = '9px IBM Plex Mono, monospace';
  ctx.textAlign = 'center';
  ctx.fillText('Warmup (2000)', tx(warmupIdx) + 36, pad.t + 12);
}

/* ── Attention Heatmap ────────────────────────────── */
const sentences = [
  { src: ['I','am','a','student','.'], tgt: ['Ich','bin','ein','Student','.'] },
  { src: ['She','is','a','teacher','.'], tgt: ['Sie','ist','eine','Lehrerin','.'] },
  { src: ['The','cat','is','small','.'], tgt: ['Die','Katze','ist','klein','.'] },
  { src: ['He','likes','music','.'], tgt: ['Er','mag','Musik','.'] },
];

function makePattern(type, sentIdx) {
  const s = sentences[sentIdx];
  const rows = type === 'cross' ? s.tgt.length :
               type === 'enc-self' ? s.src.length : s.tgt.length;
  const cols = type === 'cross' ? s.src.length :
               type === 'enc-self' ? s.src.length : s.tgt.length;

  const w = [];
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) {
      let v = 0;
      if (type === 'cross') {
        v = i === j ? 0.75 + Math.random() * 0.2
          : Math.abs(i-j) === 1 ? 0.12 + Math.random() * 0.15
          : Math.random() * 0.08;
      } else if (type === 'enc-self') {
        v = i === j ? 0.6 + Math.random() * 0.3
          : Math.random() * 0.2;
      } else {
        v = j > i ? 0 : (i === j ? 0.65 + Math.random() * 0.25 : Math.random() * 0.2);
      }
      row.push(v);
    }
    const sum = row.reduce((a,b) => a+b, 0) || 1;
    w.push(row.map(x => x/sum));
  }
  return w;
}

let currentType    = 'cross';
let currentSentIdx = 0;

function drawAttn() {
  const canvas = document.getElementById('attn-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const s   = sentences[currentSentIdx];

  const rowLabels = currentType === 'cross' ? s.tgt :
                    currentType === 'enc-self' ? s.src : s.tgt;
  const colLabels = currentType === 'cross' ? s.src :
                    currentType === 'enc-self' ? s.src : s.tgt;
  const weights   = makePattern(currentType, currentSentIdx);

  const cellW = 56, cellH = 40;
  const padL  = 80, padT  = 32;
  const W     = padL + colLabels.length * cellW + 20;
  const H     = padT + rowLabels.length * cellH + 20;

  canvas.width  = W * window.devicePixelRatio;
  canvas.height = H * window.devicePixelRatio;
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

  ctx.clearRect(0, 0, W, H);

  // Column headers
  ctx.font      = '11px IBM Plex Mono, monospace';
  ctx.fillStyle = '#475569';
  ctx.textAlign = 'center';
  colLabels.forEach((lbl, j) => {
    ctx.fillText(lbl, padL + j * cellW + cellW/2, padT - 10);
  });

  // Cells + Row headers
  rowLabels.forEach((lbl, i) => {
    ctx.textAlign = 'right';
    ctx.fillStyle = '#475569';
    ctx.font      = '11px IBM Plex Mono, monospace';
    ctx.fillText(lbl, padL - 10, padT + i * cellH + cellH/2 + 4);

    weights[i].forEach((v, j) => {
      // Vivid Indigo/Royal Blue Heatmap
      const r = Math.round(37  + (239 - 37)  * (1 - v));
      const g = Math.round(99  + (246 - 99)  * (1 - v));
      const b = Math.round(235 + (255 - 235) * (1 - v));
      
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(padL + j * cellW + 1, padT + i * cellH + 1, cellW - 2, cellH - 2);

      ctx.fillStyle = v > 0.45 ? '#FFFFFF' : '#0F172A';
      ctx.textAlign = 'center';
      ctx.font      = '9px IBM Plex Mono, monospace';
      ctx.fillText(v.toFixed(2), padL + j * cellW + cellW/2, padT + i * cellH + cellH/2 + 3);
    });
  });
}

function setAttnType(type) {
  currentType = type;
  ['cross','enc-self','dec-self'].forEach(t => {
    const btn = document.getElementById('btn-' + t);
    if (btn) btn.classList.toggle('active', t === type);
  });
  const titles = {
    'cross'    : 'Cross-attention / last decoder layer / mean across 4 heads',
    'enc-self' : 'Encoder self-attention / layer 2 / head 0',
    'dec-self' : 'Decoder self-attention / layer 2 / head 0 (causal)',
  };
  document.getElementById('attn-canvas-title').textContent = titles[type];
  drawAttn();
}

function setSentence(idx) {
  currentSentIdx = idx;
  [0,1,2,3].forEach(i => {
    const btn = document.getElementById('btn-s' + i);
    if (btn) btn.classList.toggle('active', i === idx);
  });
  drawAttn();
}

/* ── Lifecycle Initialization ────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  drawLossChart();
  drawLRChart();
  drawAttn();
});

window.addEventListener('resize', () => {
  drawLossChart();
  drawLRChart();
  drawAttn();
});