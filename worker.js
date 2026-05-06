// worker.js
// Importamos a biblioteca Transformers.js via CDN (versão 2.x)
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.0';

// Como estamos no GitHub, avisamos a IA para baixar o modelo da internet e não buscar no PC local
env.allowLocalModels = false;

// Variável para guardar o nosso modelo de IA na memória
let segmentador = null;

// Escutando as mensagens que vêm do arquivo HTML (index.html)
self.addEventListener('message', async (event) => {
    const dados = event.data;

    // 1. Comando para carregar o modelo de IA
    if (dados.acao === 'carregar') {
        self.postMessage({ status: 'iniciando' });
        
        try {
            // Carregamos o RMBG-1.4 (modelo excelente para remover fundo)
            segmentador = await pipeline('image-segmentation', 'briaai/RMBG-1.4', {
                // Função que avisa o HTML sobre a porcentagem do download do modelo
                progress_callback: (progresso) => {
                    self.postMessage({ status: 'baixando', progresso: progresso });
                }
            });
            self.postMessage({ status: 'pronto' });
        } catch (erro) {
            self.postMessage({ status: 'erro', mensagem: erro.message });
        }
    }

    // 2. Comando para processar a imagem
    if (dados.acao === 'processar') {
        if (!segmentador) return;
        
        self.postMessage({ status: 'processando' });
        
        try {
            // A IA analisa a imagem. Retorna uma imagem com fundo transparente
            const resultado = await segmentador(dados.imagemUrl);
            
            // Transformamos o resultado em algo que o HTML consegue desenhar
            // Convertendo os dados brutos para um formato de imagem (Blob)
            const canvas = new OffscreenCanvas(resultado.width, resultado.height);
            const ctx = canvas.getContext('2d');
            const imageData = new ImageData(resultado.data, resultado.width, resultado.height);
            ctx.putImageData(imageData, 0, 0);
            
            const blob = await canvas.convertToBlob({ type: 'image/png' });
            
            self.postMessage({ status: 'concluido', resultadoBlob: blob });
        } catch (erro) {
            self.postMessage({ status: 'erro', mensagem: erro.message });
        }
    }
});
