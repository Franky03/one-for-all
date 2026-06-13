# Plano de Melhorias — One for All (Hackathon)

Quatro frentes, em ordem de prioridade. Cada uma lista o objetivo, os prêmios/badges
que destrava, os detalhes técnicos e exatamente onde mexer no código.

**Decisão já tomada:** manter os 5 teachers heterogêneos (Qwen/SmolLM/Phi/Gemma/MiniCPM).
A diversidade é o que torna o gating e o fingerprint visualmente interessantes.
A ideia "all-Gemma" vira *ablation* no blog post (item 3), não configuração principal.

---

## 1. Chat ao vivo com gates por token + Arena (base vs. deku) — ✅ DONE

> **Implementado (2026-06-12):** `stream_generate` + `stream_pair` em
> `space/_probe.py` (decode manual com KV cache, gates por token com EMA,
> arena intercalada via `disable_adapter()`); `probe_fn` virou generator e
> `arena_fn` foi adicionado em `space/app.py` com a nova tab **Arena**
> (2 colunas + exemplos); `gate_html(..., ranked=False)` + transição CSS em
> `space/_html.py`; 5 testes novos em `tests/space/test_space.py`
> (33 passando). Falta só validar visualmente no Space após o push.

**Objetivo:** o usuário vê, enquanto o deku responde, qual teacher está "falando"
a cada token — e compara lado a lado a resposta do Qwen 0.5B base vs. deku.
É o que torna a destilação *sentível* em 10 segundos (critério "delightful" +
"AI load-bearing" da track Thousand Token Wood).

**Destrava:** núcleo da nota da track; matéria-prima do vídeo de demo (Best Demo, $1k).

### 1a. Streaming com gates por token

- **`space/_probe.py`** — nova função `stream_generate(text, student, tok, gating, max_new_tokens=200)`:
  loop de decodificação manual (não dá para usar `model.generate()` porque precisamos
  do hidden state a cada passo *durante* o streaming):

  ```python
  def stream_generate(text, student, tok, gating, max_new_tokens=200, ema=0.3):
      """Yields (partial_text, gate_weights) a cada token gerado."""
      prompt = tok.apply_chat_template([{"role": "user", "content": text}],
                                       tokenize=False, add_generation_prompt=True)
      enc = tok(prompt, return_tensors="pt").to(device)
      past, ids = None, enc["input_ids"]
      gate_smooth = None
      pieces = []
      for _ in range(max_new_tokens):
          with torch.no_grad():
              out = student(input_ids=ids, past_key_values=past,
                            use_cache=True, output_hidden_states=True)
          past = out.past_key_values
          h_last = out.hidden_states[-1][:, -1, :].float()      # (1, 896)
          g = gating(h_last).squeeze(0)                         # (5,)
          gate_smooth = g if gate_smooth is None else ema * g + (1 - ema) * gate_smooth
          next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
          if next_id.item() == tok.eos_token_id:
              break
          pieces.append(next_id)
          ids = next_id
          yield tok.decode(torch.cat(pieces, dim=-1)[0], skip_special_tokens=True), \
                gate_smooth.tolist()
  ```

  - EMA (`ema=0.3`) suaviza as barras — sem isso o gate por token pisca demais.
  - Manter `run_probe` como está para o ponto no UMAP (rodar **uma vez ao final**
    do streaming; `reducer.transform` é lento demais para rodar por token).

- **`space/app.py`** — transformar `probe_fn` (linha ~83) em **generator**.
  `@spaces.GPU` suporta generators no ZeroGPU; o Gradio faz streaming de cada `yield`:

  ```python
  @spaces.GPU
  def probe_fn(text, probe_points):
      ...
      for partial, gates in _probe.stream_generate(text, STUDENT, TOK, GATING):
          yield (gr.skip(), probe_points, _response_html(partial),
                 _html.gate_html(gates, VIZ["teacher_names"]), gr.skip())
      # passo final: ponto no soul space + badge de teacher dominante
      new_pt, gates = _probe.run_probe(text, STUDENT, TOK, GATING, REDUCER)
      updated = probe_points + [new_pt]
      yield (_glb.build_glb(VIZ, COORDS3D, updated), updated,
             _response_html(partial), _html.gate_html(gates, ...), _html.task_html(gates, ...))
  ```

  (`gr.skip()` evita re-renderizar o Model3D a cada token — só atualiza no fim.)

- **`space/_html.py`** — em `gate_html` (linha ~22), adicionar
  `transition:width 0.18s ease` no `div` interno da barra. Como o HTML é
  recriado a cada yield, **remover o `sorted(...)`** durante o streaming (ordem
  fixa pelos índices) — senão as barras trocam de lugar e a animação quebra.
  Sugestão: `gate_html(..., ranked: bool = True)` e passar `ranked=False` no streaming.

### 1b. Arena — base vs. deku na mesma pergunta

- **Truque-chave: zero memória extra.** O student é um `PeftModel`; a resposta do
  modelo base sai do mesmo objeto com o adapter desligado:

  ```python
  with student.disable_adapter():
      base_answer = generate_response(text, student, tok)
  ```

  Nada de carregar um segundo modelo.

- **`space/_probe.py`** — `generate_pair(text, ...)`: gera base (adapter off) e
  deku (adapter on), idealmente intercalando os dois streamings para efeito
  "corrida lado a lado".
- **`space/app.py`** — novo bloco na Tab Souls (ou tab "Arena"): `gr.Row` com duas
  colunas (`BASE · Qwen2.5-0.5B` / `DEKU · distilled`), um único textbox, e abaixo
  de cada coluna o tempo de resposta. Reusar `_response_html` com um parâmetro de título.
- Adicionar 3–4 prompts de exemplo (`gr.Examples`) que mostram diferença real —
  escolher com base no benchmark (GSM8K/ARC onde o delta é positivo).

### Validação

- `tests/space/test_space.py`: teste de `stream_generate` com um modelo dummy
  (2 layers, vocab pequeno) verificando que yields são monotônicos e gates somam 1.
- Local: `cd space && VIZ_DATA_PATH=... python app.py` (sem GPU funciona em CPU para 0.5B).

---

## 2. GGUF + llama.cpp (badge 🦙 Llama Champion) — ✅ DONE (código)

> **Implementado (2026-06-12):** `export_gguf` em `modal_app.py` (imagem CPU
> própria com llama.cpp; merge LoRA → f16 + q8_0 via `convert_hf_to_gguf.py`
> direto, sem cmake; `gating.npz` para gate em numpy; `--push` opcional para
> `build-small-hackathon/deku-gguf`). Backend `OFA_BACKEND=llamacpp` em
> `space/_probe.py` (`LlamaCppStudent`: geração streaming + embedding mean-pool
> para gate/UMAP; Arena via GGUF oficial do base Qwen, lazy). `app.py` despacha
> por backend; pill 🦙 no header. **Pendente (operacional):** rodar
> `modal run ofa/modal_app.py::export_gguf --push` após o retrain e setar
> `OFA_BACKEND=llamacpp` no Space para o modo CPU.
> **Bônus na mesma leva:** Nemotron-Mini-4B-Instruct adicionado como 6º teacher
> (`config.py`, hidden 3072) → candidatura ao NVIDIA Nemotron Quest; paletas,
> header e smoke tests atualizados para 6; o Space agora infere n_teachers do
> checkpoint `gating.pt`, então continua compatível com o deku atual (5) até o
> retrain sair.

**Objetivo:** o student rodando via runtime llama.cpp. Bônus: derruba a dependência
de GPU do Space (0.5B Q8_0 roda bem em CPU) → reforça também o badge 🔌 Off the Grid.

**Destrava:** Llama Champion; Off the Grid; soma para Bonus Quest Champion ($2k).

### 2a. Exportar GGUF (uma vez, no Modal)

- **`modal_app.py`** — nova função `export_gguf`:
  1. Carregar base + adapter do volume (`/vol/ofa_student/final`), `merged = PeftModel.from_pretrained(...).merge_and_unload()`,
     `merged.save_pretrained("/tmp/deku-merged")` + tokenizer.
  2. `git clone https://github.com/ggerganov/llama.cpp` na imagem (ou adicionar ao
     `Image.run_commands`), `pip install -r llama.cpp/requirements.txt`.
  3. `python llama.cpp/convert_hf_to_gguf.py /tmp/deku-merged --outfile deku-f16.gguf --outtype f16`
  4. `llama-quantize deku-f16.gguf deku-q8_0.gguf Q8_0` (e `Q4_K_M` para a versão mínima).
  5. Upload para o Hub: repo **`build-small-hackathon/deku-gguf`** (model card citando o repo principal).
- Custo: CPU-only no Modal serve (sem A100).

### 2b. Backend llama.cpp no Space

- **`space/_probe.py`** — flag `OFA_BACKEND` (env): `"torch"` (default) ou `"llamacpp"`.
  - Geração: `llama_cpp.Llama(model_path=hf_hub_download("build-small-hackathon/deku-gguf", "deku-q8_0.gguf"), n_ctx=1024)`;
    streaming nativo via `create_chat_completion(..., stream=True)` — encaixa direto
    no item 1.
  - Gates: o `GatingNetwork` é só um `Linear(896→5)`. Exportar pesos para
    `gating.npz` (junto do `gating.pt` no repo deku) e calcular o gate em **numpy**:
    `softmax(emb @ W.T + b)`. Embedding por token vem do llama.cpp com
    `Llama(..., embedding=True)` (Qwen2.5-0.5B → 896-dim, mesma dimensão do torch path).
  - Aceitar pequena deriva numérica vs. torch (quantização Q8_0) — mencionar no blog.
- **`space/requirements.txt`** — adicionar `llama-cpp-python` (wheel CPU).
- **`space/README.md`** — documentar o badge: "student runs through llama.cpp (Q8_0)".
- Fallback honesto: se o embedding per-token do llama.cpp der trabalho, manter
  gates no torch e **geração** via llama.cpp — o badge pede que o modelo rode no
  runtime llama.cpp, e a geração é o modelo rodando.

### Validação

- Script local rápido: baixar o Q8_0 e comparar 5 respostas vs. torch path.
- Smoke: `OFA_BACKEND=llamacpp python space/app.py` em CPU.

---

## 3. Field Notes (blog post) + ablation all-Gemma

**Objetivo:** post técnico no HF (community blog) contando o que foi construído e
**o que foi aprendido** — incluindo a ablation "teachers diversos vs. mesma família",
que transforma a ideia do all-Gemma em ciência publicável em vez de mudança de arquitetura.

**Destrava:** badge 📓 Field Notes; material para o social post (obrigatório);
visibilidade para OpenBMB Awards (MiniCPM como teacher) e Modal Awards (treino no Modal).

### 3a. Suporte a conjuntos de teachers alternativos

- **`config.py`** — ao lado de `DEFAULT_TEACHERS` (linha 29), adicionar:

  ```python
  GEMMA_TEACHERS: list[TeacherSpec] = [
      TeacherSpec("gemma-1b", "google/gemma-3-1b-it", 1152, "same family, 1B"),
      TeacherSpec("gemma-2b", "google/gemma-2-2b-it", 2304, "same family, 2B"),
      TeacherSpec("gemma-4b", "google/gemma-3-4b-it", 2560, "same family, 4B"),
  ]
  TEACHER_SETS = {"diverse": DEFAULT_TEACHERS, "gemma": GEMMA_TEACHERS}
  ```

  (Conferir `hidden_dim` de cada um no `config.json` do Hub antes de rodar.
  Atenção: sem teacher Qwen no set gemma não há par same-tokenizer → treinar
  com `lambda_kd_max=0.0`, geometry-only puro. Isso é parte do achado da ablation.)

- **`modal_app.py`** — parâmetro `--teacher-set diverse|gemma` no entrypoint de
  treino, mapeando para `TEACHER_SETS`; nome do checkpoint no volume sufixado
  (`final_gemma/`) para não sobrescrever o run principal.
- **`models/loader.py`** e **`distill/trainer.py`** — já são genéricos sobre
  `cfg.teachers`; só conferir que `N_TEACHERS` não está hardcoded em nenhum ponto
  do caminho de treino (no Space está: `space/_probe.py` linha 11 — ok, Space só
  usa o run principal).

### 3b. Rodar e medir

1. Run principal (diverse, 5k steps) — já existe / re-treinar se necessário.
2. Run ablation: `modal run ofa/modal_app.py --steps 5000 --teacher-set gemma`.
3. `modal run ofa/modal_app.py::benchmark` para os dois (gsm8k, arc_challenge, hellaswag).
4. Comparar: tabela de deltas, matriz CKA (diverse deve ter off-diagonal mais baixo),
   entropia do gate ao longo do treino (hipótese: no set gemma o gate colapsa no maior teacher).

### 3c. Escrever e publicar

- **`docs/field_notes.md`** — estrutura sugerida:
  1. O problema: 5 tokenizers, 1 student (por que KL não funciona cross-vocab → Path B)
  2. Gated CKA: a matemática em 1 parágrafo + diagrama
  3. O curriculum de 3 fases (`LossSchedule`) e por que warmup importa
  4. **Ablation: teachers diversos vs. all-Gemma** — números + leitura
  5. Benchmarks honestos (deltas, incluindo onde não melhorou)
  6. O que eu faria diferente
- Publicar como community article em `huggingface.co/blog` (botão "New article"),
  linkar no Space README e no model card do deku.

---

## 4. UI custom com `gr.Server` (badge 🎨 Off-Brand + Off-Brand Award $1.5k) — ✅ DONE (código)

> **Implementado (2026-06-12):** startup compartilhado extraído para
> `space/_boot.py` (Runtime + dispatch torch/llamacpp + `viz_payload`);
> `space/server_app.py` com `gr.Server` (gradio 6.18): endpoints `viz`,
> `probe` e `arena` (generators streamam via SSE — **gotcha descoberto:
> generators precisam anotar o tipo do chunk no retorno**, ex. `-> dict`,
> senão `get_return_types` infere zero outputs e `data` vem vazio);
> frontend custom em `space/frontend/` (index.html + style.css + app.js):
> Three.js com orbit/zoom manual, pontos como sprites aditivos, probe points
> pulsando ao nascer, console com tabs PROBE/ARENA, quirk meters animados,
> tema próprio "soul transfer console". Validado local: rotas 200, traversal
> bloqueado, SSE de generator e não-generator OK, `app.py` legado também
> constrói sob gradio 6.18 (fallback real). Frontmatter do Space trocado para
> `app_file: server_app.py` + `sdk_version: 6.18.0`. **Rollback:** voltar
> frontmatter para `app.py` (funciona no mesmo sdk). **Pendente:** validar no
> Spaces com ZeroGPU (o `@gradio/client` do browser já é o caminho exigido).

**Objetivo:** sair do layout Blocks e servir um frontend próprio (HTML/JS/Three.js)
com o Gradio como backend — exatamente a dica do hackathon ("see `gr.Server`").

**Como o `gr.Server` funciona** (confirmado na doc oficial): é um FastAPI com a
engine de API do Gradio embutida. Funções viram endpoints com `@app.api(name=...)`
(generators fazem streaming via SSE automaticamente), e você serve seu HTML em `/`
com rotas FastAPI normais. O browser chama os endpoints via `@gradio/client` JS —
**obrigatório quando se usa ZeroGPU** (encaminha os headers de quota).

### Estrutura

```
space/
  server_app.py          # gr.Server: endpoints + rota / servindo o frontend
  frontend/
    index.html           # layout custom (tema "soul transfer" — verde OFA/raios)
    app.js               # @gradio/client + Three.js
    style.css
```

### `space/server_app.py`

```python
from gradio import Server
from fastapi.responses import HTMLResponse, FileResponse
import spaces

app = Server()

@app.api(name="probe")
@spaces.GPU
def probe(text: str, probe_points: list):
    for partial, gates in _probe.stream_generate(...):   # reusa item 1
        yield {"text": partial, "gates": gates}
    yield {..., "point": new_pt}                          # ponto UMAP no final

@app.api(name="arena")
@spaces.GPU
def arena(text: str): ...                                 # base vs deku (item 1b)

@app.api(name="viz")
def viz():                                                # dados estáticos p/ Three.js
    return {"coords": COORDS3D.tolist(), "cka": VIZ["cka"], "curves": VIZ["curves"]}

@app.get("/", response_class=HTMLResponse)
async def home():
    return open("frontend/index.html").read()

app.launch()
```

### Frontend (`frontend/app.js`)

- `import { Client } from "@gradio/client"` (CDN) → `client.submit("/probe", ...)`
  e iterar os eventos SSE para animar texto + barras de gate.
- Soul space em **Three.js puro** (esferas coloridas por modelo, ponto do probe
  caindo com animação) — substitui o `gr.Model3D`/GLB; reaproveitar a lógica de
  cores/legenda de `space/_glb.py` e `space/_three.py`.
- Tema: identidade própria (não-Gradio): tipografia display, fundo com energia
  do One for All, gate bars como "quirk meters". O CSS atual de `app.py`
  (linhas 103–292) já tem a paleta — migrar para `style.css`.

### Migração segura

1. Desenvolver `server_app.py` com `app.py` intacto (são arquivos separados).
2. Testar local: `python space/server_app.py` (endpoints aparecem em `/gradio_api/docs`).
3. Trocar `app_file: app.py` → `app_file: server_app.py` no frontmatter de
   **`space/README.md`** (linha 8) e subir `sdk_version` para a versão de Gradio
   que inclui `gr.Server` (5.x recente — conferir changelog ao implementar;
   atualizar `space/requirements.txt` junto).
4. Manter screenshot/GIF antigo e novo no README — "antes/depois" é ótimo
   conteúdo de social post.

---

## Ordem de execução sugerida

| # | Item | Esforço | Dependências |
|---|------|---------|--------------|
| 1 | ✅ Streaming + gates por token + Arena | ~1 dia | nenhuma |
| 2 | ✅ GGUF + llama.cpp (+ Nemotron 6º teacher) | ~½ dia | falta retrain + export no Modal |
| 3a/3b | Ablation all-Gemma (treino + bench) | ~½ dia de código + horas de GPU Modal | nenhuma |
| 4 | ✅ gr.Server frontend | 1–2 dias | validar no Spaces após push |
| 3c | Blog post | ~½ dia | resultados de 2 e 3b |

## Checklist de prêmios cobertos ao final

- ✅ 🎯 Well-Tuned (deku publicado) — já tem
- ✅ 🐜 Tiny Titan — student 0.5B; **destacar no Space README e no vídeo**
- 🔲 🦙 Llama Champion — item 2
- 🔲 📓 Field Notes — item 3
- 🔲 🎨 Off-Brand (badge + award) — item 4
- 🔲 🔌 Off the Grid — itens 2 + 4 (inferência 100% local no Space, documentar)
- 🔲 🎖️ Bonus Quest Champion — consequência dos acima
- 🔲 🎬 Best Demo — vídeo gravado em cima do item 1 (Arena + gates ao vivo)
- Sponsor: OpenBMB (MiniCPM teacher — citar explicitamente), Modal (treino/export — citar), NVIDIA (✅ Nemotron-Mini-4B-Instruct é o 6º teacher — citar no Space/blog para o Nemotron Quest)
