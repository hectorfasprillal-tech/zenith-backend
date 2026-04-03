import os
import re
import unicodedata
from .semantic_search import search
from sentence_transformers import CrossEncoder
from langchain_core.prompts import PromptTemplate
try:
    from langchain_ollama import OllamaLLM
except Exception:
    try:
        from langchain_community.llms import Ollama as OllamaLLM
    except Exception:
        OllamaLLM = None

_cross_model = None

def get_cross_model():
    global _cross_model
    if _cross_model is None:
        try:
            _cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            print(f"⚠️ RERANK WARN: CrossEncoder no disponible: {str(e)}")
            _cross_model = None
    return _cross_model

def get_llm():
    model_name = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')
    try:
        if OllamaLLM is None:
            return None
        return OllamaLLM(model=model_name, temperature=float(os.getenv('LLM_TEMPERATURE', '0.2')))
    except Exception as e:
        print(f"⚠️ LLM WARN: Ollama no disponible: {str(e)}")
        return None


def build_doc_text(md):
    return ' '.join([
        normalize_text(md.get('Providencia', '')),
        normalize_text(md.get('Información', '')),
        normalize_text(md.get('summary_extract', '')),
        normalize_text(md.get('Resumen', '')),
        normalize_text(md.get('Tema - subtema', '')),
        normalize_text(md.get('Tema', '')),
    ]).strip()
_cross_model = None

def get_cross_model():
    global _cross_model
    if _cross_model is None:
        try:
            _cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            print(f"⚠️ RERANK WARN: CrossEncoder no disponible: {str(e)}")
            _cross_model = None
    return _cross_model


def build_doc_text(md):
    return ' '.join([
        normalize_text(md.get('Información', '')),
        normalize_text(md.get('summary_extract', '')),
        normalize_text(md.get('Resumen', '')),
        normalize_text(md.get('Tema - subtema', '')),
        normalize_text(md.get('Tema', '')),
    ]).strip()

def html_escape(s):
    if s is None:
        return 'N/A'
    s = str(s)
    s = s.replace('&', '&amp;')
    s = s.replace('<', '&lt;')
    s = s.replace('>', '&gt;')
    s = s.replace('"', '&quot;')
    s = s.replace("'", '&#39;')
    return s

def normalize_text(text):
    if not text:
        return ''
    text = unicodedata.normalize('NFC', str(text))
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fix_typos(text):
    t = text or ''
    t = re.sub(r'\bsobree\b', 'sobre', t, flags=re.IGNORECASE)
    t = re.sub(r'\baboro\b', 'aborto', t, flags=re.IGNORECASE)
    t = re.sub(r'interrupcion\s*emb\.?', 'interrupcion voluntaria del embarazo', t, flags=re.IGNORECASE)
    return t

def _keyword_tokens(s):
    s = normalize_text(fix_typos(s))
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()
    base = set(re.findall(r'\b[\w]{3,}\b', s))
    if {'ive','interrupcion','aborto'} & base:
        base |= {'interrupcion','voluntaria','embarazo','aborto','ive'}
    if 'eutanasia' in base:
        base |= {'eutanasia','muerte','digna'}
    return base

def expand_queries(prompt):
    p_norm = normalize_text(prompt)
    p_noacc = ''.join(c for c in unicodedata.normalize('NFD', p_norm) if unicodedata.category(c) != 'Mn')
    tokens = _keyword_tokens(prompt)
    variants = {p_norm, p_noacc}
    if {'ive', 'interrupcion'} & tokens or 'aborto' in tokens:
        variants |= {
            "IVE", "interrupción voluntaria del embarazo", "interrupcion voluntaria del embarazo", "aborto",
            "despenalización del aborto", "tres causales", "derechos reproductivos", "salud sexual y reproductiva",
            "C-355", "C-355/06", "C 355/06", "C355/06", "C-355 06", "C 355 de 2006", "Sentencia C-355 de 2006",
            "C-055", "C-055/22", "C 055/22", "C055/22", "C 055 de 2022", "Sentencia C-055 de 2022",
            "C-754-15", "Sentencia C-754 de 2015"
        }
    if 'libertad' in tokens and 'condicional' in tokens:
        variants |= {"libertad condicional", "beneficios penitenciarios", "sustitución de pena"}
    return list(variants)

def derive_filters(tokens):
    filt = {}
    tema_aborto = {'aborto', 'ive', 'interrupcion'} & tokens
    incluye_tutela = 'tutela' in tokens
    if tema_aborto and not incluye_tutela:
        filt['tipo'] = 'Constitucionalidad'
    if '355' in tokens:
        filt['anio'] = 2006
        if not incluye_tutela:
            filt['tipo'] = 'Constitucionalidad'
    if '055' in tokens:
        filt['anio'] = 2022
        if not incluye_tutela:
            filt['tipo'] = 'Constitucionalidad'
    return filt

def simple_keyword_boost(results, prompt, k=20):
    tokens = _keyword_tokens(prompt)
    scored = []
    for r in results:
        md = r.get('metadata', {})
        doc_raw = build_doc_text(md)
        doc = ''.join(c for c in unicodedata.normalize('NFD', doc_raw) if unicodedata.category(c) != 'Mn').lower()
        hits = sum(1 for t in tokens if t in doc)
        prov = normalize_text(md.get('Providencia', '')).upper()
        tema_all = normalize_text(md.get('Tema - subtema', '') + ' ' + md.get('Tema', '')).lower()
        meta_boost = 0.0
        if 'C-355' in prov or 'C-055' in prov:
            meta_boost += 2.0
        if any(t in tema_all for t in ['aborto','interrupcion','ive']):
            meta_boost += 1.5
        if any(t in doc for t in ['aborto','interrupcion','ive']):
            meta_boost += 1.0
        combined = float(r.get('score', 0)) + hits * 0.6 + meta_boost
        scored.append((combined, r, hits, meta_boost))
    scored.sort(key=lambda x: x[0], reverse=True)
    filtered = [r for _, r, h, mb in scored if h > 0 or mb > 0]
    if len(filtered) == 0:
        filtered = [r for _, r, _, _ in scored]
    return filtered[:k]

def extract_snippets(metadata, tokens, max_snippets=4):
    fields = [
        metadata.get('Información', ''),
        metadata.get('summary_extract', ''),
        metadata.get('Resumen', ''),
        metadata.get('Tema - subtema', ''),
        metadata.get('Tema', ''),
    ]
    text = ' '.join([normalize_text(f or '') for f in fields]).strip()
    if not text:
        return []
    sents = re.split(r'(?<=[\.!?])\s+', text)
    hits = []
    for s in sents:
        s_norm = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()
        if any(t in s_norm for t in tokens):
            hits.append(truncate(s, limit=300))
            if len(hits) >= max_snippets:
                break
    return hits

def looks_gibberish(text):
    if not text:
        return True
    t = text
    # Patterns of junk content
    if re.search(r'(bjbj\s*){2,}', t, flags=re.IGNORECASE):
        return True
    if re.search(r'(?:\b[A-Za-zÁÉÍÓÚÜÑ]\b\s*){6,}', t):
        return True
    if re.search(r'(?:\b[A-Za-zÁÉÍÓÚÜÑ]{1,2}\b\s*){12,}', t):
        return True
    # Quality heuristics
    letters = sum(c.isalpha() for c in t)
    spaces = t.count(' ')
    ratio = (letters + spaces) / max(len(t), 1)
    long_words = len([w for w in re.findall(r'\b\w+\b', t) if len(w) >= 5])
    return ratio < 0.6 or long_words < 2 or len(t) < 40

def clean_summary(text):
    t = normalize_text(text)
    # Remove repeated junk patterns
    t = re.sub(r'(bjbj\s*){1,}', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'(?:\b[A-Za-zÁÉÍÓÚÜÑ]\b\s*){6,}', ' ', t)
    t = re.sub(r'(?:\b[A-Za-zÁÉÍÓÚÜÑ]{1,2}\b\s*){12,}', ' ', t)
    # Prefer first two sentences
    sentences = re.split(r'(?<=[.!?])\s+', t)
    t2 = ' '.join(sentences[:2]).strip()
    if len(t2) < 80:
        cutoff = 350
        if len(t) > cutoff:
            last_period = t.rfind('.', 0, cutoff)
            if last_period >= 200:
                t2 = t[:last_period+1]
            else:
                t2 = t[:cutoff]
        else:
            t2 = t
    return t2.strip()

def truncate(text, limit=120):
    t = normalize_text(text)
    if len(t) <= limit:
        return t
    return t[:limit].rstrip() + '…'

def choose_summary(metadata):
    preferred = metadata.get('summary_extract')
    fallback = metadata.get('Resumen')
    info = metadata.get('Información')
    for candidate in [preferred, fallback, info]:
        candidate_clean = clean_summary(candidate) if candidate else ""
        if candidate_clean and not looks_gibberish(candidate_clean):
            return candidate_clean
    tema = metadata.get('Tema - subtema', metadata.get('Tema', ''))
    tema_clean = normalize_text(tema)
    if tema_clean:
        return f"Tema principal: {tema_clean}"
    tipo = metadata.get('Tipo', '')
    providencia = metadata.get('Providencia', '')
    return f"Sentencia {normalize_text(providencia)} ({normalize_text(tipo)})"

def has_topic_coverage(results, tokens):
    for r in results:
        md = r.get('metadata', {})
        doc_raw = build_doc_text(md)
        doc = ''.join(c for c in unicodedata.normalize('NFD', doc_raw) if unicodedata.category(c) != 'Mn').lower()
        if any(t in doc for t in tokens):
            return True
    return False

def filter_by_required_tokens(results, required):
    if not required:
        return results
    kept = []
    for r in results:
        md = r.get('metadata', {})
        doc_raw = build_doc_text(md)
        doc = ''.join(c for c in unicodedata.normalize('NFD', doc_raw) if unicodedata.category(c) != 'Mn').lower()
        if any(t in doc for t in required):
            kept.append(r)
    return kept

def classify_intent(tokens):
    topic = 'general'
    required = set()
    if {'aborto','ive','interrupcion'} & tokens:
        topic = 'aborto'
        required = {'aborto','ive','interrupcion'}
    elif 'eutanasia' in tokens:
        topic = 'eutanasia'
        required = {'eutanasia'}
    # Añade numerales de providencias para que el filtrado por tema considere C-355/06 y C-055/22
    if '355' in tokens:
        required |= {'355'}
    if '055' in tokens:
        required |= {'055'}
    action = 'summary'
    if {'diferencias','comparacion','comparar','compara'} & tokens:
        action = 'compare'
    elif {'evolucion','resumen','resume'} & tokens:
        action = 'chronology'
    elif {'recientes','reciente','actuales','mas'} & tokens:
        action = 'recent'
    return {'topic': topic, 'action': action, 'required': required}

def ensure_topic_candidates(combined_results, required_tokens):
    filtered = []
    for r in combined_results:
        md = r.get('metadata', {})
        doc_raw = build_doc_text(md)
        doc = ''.join(c for c in unicodedata.normalize('NFD', doc_raw) if unicodedata.category(c) != 'Mn').lower()
        if not required_tokens or any(t in doc for t in required_tokens):
            filtered.append(r)
    return filtered if filtered else combined_results

def topic_anchor_queries(topic):
    if topic == 'aborto':
        return [
            "C-355/06","C 355 de 2006","Sentencia C-355 de 2006",
            "C-055/22","C 055 de 2022","Sentencia C-055 de 2022",
            "C-754-15","Sentencia C-754 de 2015"
        ]
    return []

def topic_anchor_queries(topic):
    if topic == 'aborto':
        return [
            "C-355/06","C 355 de 2006","Sentencia C-355 de 2006",
            "C-055/22","C 055 de 2022","Sentencia C-055 de 2022",
            "C-754-15","Sentencia C-754 de 2015"
        ]
    return []

def chatbot(prompt):
    print(f"\n🔍 DEBUG: Pregunta recibida: {prompt}")
    try:
        variants = expand_queries(prompt)
        print(f"🔍 DEBUG: Consultas variantes: {variants}")
        tokens = _keyword_tokens(prompt)
        intent = classify_intent(tokens)
        filt = derive_filters(tokens)
        if filt:
            print(f"🔍 DEBUG: Filtros derivados: {filt}")
        print(f"🔍 DEBUG: Intent derivado: {intent}")
        combined = {}
        for q in variants:
            payload = {'text': q, 'top_k': max(10, 35 // max(len(variants),1))}
            payload.update(filt)
            sr = search(payload)
            for m in sr['matches']:
                mid = m.get('id') or m.get('metadata', {}).get('id') or str(m)
                if mid not in combined or float(m.get('score', 0)) > float(combined[mid].get('score', 0)):
                    combined[mid] = m
        results = list(combined.values())
        print(f"🔍 DEBUG: Número de resultados combinados: {len(results)}")
        if not has_topic_coverage(results, tokens):
            print("🔍 DEBUG: Cobertura temática insuficiente; segunda pasada de recuperación...")
            alt = [
                "despenalización del aborto","tres causales","IVE",
                "interrupción voluntaria del embarazo","C-355/06","C-055/22",
                "salud sexual y reproductiva","derechos reproductivos","protocolo IVE","objeción de conciencia IVE",
                "Sentencia C-754 de 2015"
            ]
            for q in alt:
                payload = {'text': q, 'top_k': 35}
                sr = search(payload)
                for m in sr['matches']:
                    mid = m.get('id') or m.get('metadata', {}).get('id') or str(m)
                    if mid not in combined or float(m.get('score', 0)) > float(combined[mid].get('score', 0)):
                        combined[mid] = m
            results = list(combined.values())
            print(f"🔍 DEBUG: Número de resultados tras segunda pasada: {len(results)}")
        results = ensure_topic_candidates(results, intent.get('required', set()))
        print(f"🔍 DEBUG: Resultados tras filtrado por tema: {len(results)}")
        anchors = topic_anchor_queries(intent.get('topic'))
        if anchors and not has_topic_coverage(results, intent.get('required', set())):
            print(f"🔍 DEBUG: Inyección de anclas por tema: {anchors}")
            for q in anchors:
                payload = {'text': q, 'top_k': 20, 'tipo': 'Constitucionalidad'}
                sr = search(payload)
                for m in sr['matches']:
                    mid = m.get('id') or m.get('metadata', {}).get('id') or str(m)
                    if mid not in combined or float(m.get('score', 0)) > float(combined[mid].get('score', 0)):
                        combined[mid] = m
            results = ensure_topic_candidates(list(combined.values()), intent.get('required', set()))
            print(f"🔍 DEBUG: Resultados tras anclas y filtrado: {len(results)}")
        anchors = topic_anchor_queries(intent.get('topic'))
        if anchors and not has_topic_coverage(results, intent.get('required', set())):
            print(f"🔍 DEBUG: Inyección de anclas por tema: {anchors}")
            for q in anchors:
                payload = {'text': q, 'top_k': 20, 'tipo': 'Constitucionalidad'}
                sr = search(payload)
                for m in sr['matches']:
                    mid = m.get('id') or m.get('metadata', {}).get('id') or str(m)
                    if mid not in combined or float(m.get('score', 0)) > float(combined[mid].get('score', 0)):
                        combined[mid] = m
            results = ensure_topic_candidates(list(combined.values()), intent.get('required', set()))
            print(f"🔍 DEBUG: Resultados tras anclas y filtrado: {len(results)}")
    except Exception as e:
        print(f"❌ ERROR en búsqueda: {str(e)}")
        return f"Error en la búsqueda: {str(e)}"
    if not results:
        print("⚠️ DEBUG: No se encontraron resultados")
        return "No encontré información relevante en la base de datos de sentencias."
    model = get_cross_model()
    candidate_results = results
    top_candidates = simple_keyword_boost(candidate_results, prompt, k=30)
    tc_req = filter_by_required_tokens(top_candidates, intent.get('required', set()))
    if tc_req:
        top_candidates = tc_req
    pairs = []
    texts = []
    for r in top_candidates:
        md = r.get('metadata', {})
        t = build_doc_text(md)
        if not t:
            t = normalize_text(md.get('Información', ''))
        texts.append(t)
        pairs.append((prompt, t))
    reranked = top_candidates
    try:
        if model:
            scores = model.predict(pairs)
            scored = list(zip(scores, top_candidates))
            scored.sort(key=lambda x: float(x[0]), reverse=True)
            reranked = [r for _, r in scored[:8]]
        else:
            print("⚠️ RERANK WARN: usando orden por score del embedding")
            reranked = sorted(top_candidates, key=lambda r: float(r.get('score', 0)), reverse=True)[:8]
    except Exception as e:
        print(f"⚠️ RERANK WARN: fallo en reranker: {str(e)}")
        reranked = sorted(top_candidates, key=lambda r: float(r.get('score', 0)), reverse=True)[:8]
    contexto = "Información relevante de sentencias:\n\n"
    kw = _keyword_tokens(prompt)
    for i, result in enumerate(reranked, 1):
        metadata = result['metadata']
        print(f"🔍 DEBUG: Metadata de resultado {i}: {metadata}")
        contexto += f"Sentencia {i}:\n"
        contexto += f"- Providencia: {metadata.get('Providencia', 'N/A')}\n"
        contexto += f"- Tema: {metadata.get('Tema - subtema', metadata.get('Tema', 'N/A'))}\n"
        contexto += f"- Fecha: {metadata.get('fecha', metadata.get('anio', 'N/A'))}\n"
        contexto += f"- Tipo: {metadata.get('Tipo', 'N/A')}\n"
        frags = extract_snippets(metadata, kw, max_snippets=5)
        if frags:
            contexto += "- Fragmentos:\n" + "\n".join([f"  • {f}" for f in frags]) + "\n\n"
        else:
            contexto += f"- Resumen: {metadata.get('summary_extract', metadata.get('Resumen', metadata.get('Información', 'N/A')))}\n\n"
    print(f"🔍 DEBUG: Contexto construido:\n{contexto}")
    prompt_completo = f"""Basándote en la siguiente información de sentencias judiciales, responde la pregunta en español de manera clara y concisa.

{contexto}

Pregunta: {prompt}

Respuesta en español:"""
    print(f"🔍 DEBUG: Prompt completo enviado al modelo:\n{prompt_completo}")
    llm = get_llm()
    if llm:
        print("🔍 DEBUG: Generando respuesta con LLM...")
        template = PromptTemplate(
            input_variables=["contexto", "pregunta"],
            template=(
                "Eres un asistente que resume jurisprudencia pública de la Corte Constitucional de Colombia. "
                "Responde en HTML, en español, citando providencias. No des asesoramiento legal ni te niegues a responder. "
                "Usa exclusivamente la información del contexto; si algo no consta, di: 'no consta en el contexto'. "
                "No uses 'lo siento', 'no hay información', 'no puedo', 'no tengo acceso', ni 'asesoramiento legal'. Sé claro y conciso, sin disculpas ni advertencias.\n\n"
                "Contexto:\n{contexto}\n\nPregunta: {pregunta}\n\nRespuesta en HTML:"
            ),
        )
        try:
            prompt_llm = template.format(contexto=contexto, pregunta=prompt)
            respuesta_llm = llm.invoke(prompt_llm)
            resp = respuesta_llm if isinstance(respuesta_llm, str) else str(respuesta_llm)
            if re.search(r'(lo siento|no hay información|no consta|no puedo)', resp, flags=re.IGNORECASE):
                print("⚠️ LLM WARN: salida con disclaimer; generando resumen basado en contexto")
                respuesta = (
                    f"<p>Basándome en las sentencias encontradas sobre '{html_escape(prompt)}', "
                    f"te puedo informar lo siguiente:</p><ol>"
                )
                for i, result in enumerate(reranked, 1):
                    metadata = result['metadata']
                    providencia = html_escape(metadata.get('Providencia', 'N/A'))
                    tema = truncate(metadata.get('Tema - subtema', metadata.get('Tema', 'N/A')))
                    tema = html_escape(tema)
                    resumen = html_escape(choose_summary(metadata))
                    respuesta += (
                        f"<li><p><strong>Sentencia {providencia}</strong> "
                        f"<em>(Tema: {tema})</em></p>"
                        f"<p>{resumen}</p></li>"
                    )
                respuesta += ("</ol><p>Esta información proviene de la jurisprudencia constitucional colombiana.</p>")
                print(f"🔍 DEBUG: Respuesta generada (sanitizada): {respuesta}")
                return respuesta
            return resp
        except Exception as e:
            print(f"⚠️ LLM WARN: fallo al generar con LLM: {str(e)}")
    try:
        print("🔍 DEBUG: Generando síntesis concisa por fallback...")
        refs = []
        seen = set()
        for r in reranked:
            md = r.get('metadata', {})
            p = md.get('Providencia', '').strip()
            if p and p not in seen:
                refs.append(p)
                seen.add(p)
            if len(refs) >= 3:
                break
        citations = ', '.join([html_escape(x) for x in refs])
        topic_label = 'aborto/IVE' if intent.get('topic') == 'aborto' else 'el tema consultado'
        respuesta = (
            f"<p>Según la jurisprudencia constitucional, {('las sentencias ' + citations) if citations else 'las providencias consultadas'} "
            f"abordan {topic_label}.</p>"
        )
        print(f"🔍 DEBUG: Respuesta generada (fallback concisa): {respuesta}")
        return respuesta
    except Exception as e:
        print(f"❌ ERROR al generar respuesta: {str(e)}")
        return f"Error al generar respuesta: {str(e)}"
