# Editor de Imagens IA — Remoção de Fundo

Ferramenta local de remoção de fundo para imagens de produtos (e-commerce).  
Funciona 100% offline — sem depender de CDNs externas bloqueadas por redes corporativas.

## Como funciona

O servidor detecta automaticamente o tipo de fundo da imagem:

| Tipo de Imagem | Técnica Usada | Resultado |
|---|---|---|
| **Fundo branco** (produtos e-commerce) | Remoção por cor (threshold) | Preserva correntes, brincos, detalhes finos |
| **Fundo lifestyle** (pessoas, ambientes) | IA rembg (segmentação neural) | Remove o fundo complexo automaticamente |

## Requisitos

- Python 3.10 ou superior
- Conexão com internet apenas na **primeira vez** (para baixar o modelo de IA ~170MB)

## Instalação

```bash
pip install -r requirements.txt
```

## Como usar

```bash
python servidor_ia.py
```

O servidor inicia em `http://localhost:5001` e abre o navegador automaticamente.  
Arraste ou selecione imagens na interface para remover o fundo.

## Estrutura do projeto

```
├── servidor_ia.py      # Servidor Flask local com a lógica de processamento
├── index.html          # Interface web (drag & drop)
├── requirements.txt    # Dependências Python
└── README.md
```

## Detalhes técnicos

- **Fundo branco**: calcula a distância euclidiana de cada pixel ao branco puro. Pixels dentro do limiar viram transparentes. Bordas são suavizadas com GaussianBlur para evitar serrilhado.
- **Fundo complexo**: usa o modelo `isnet-general-use` do `rembg` com `alpha_matting` ativado para preservar bordas finas.
