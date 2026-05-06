// worker.js
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers';

env.allowLocalModels = false;
let segmentador = null;

self.addEventListener('message', async (event) => {
    const dados = event.data;

    // 1. Comando para carregar o modelo de IA
    if (dados.acao === 'carregar') {
        self.postMessage({ status: 'iniciando' });
        
        try {
            // Combinação correta: Gaveta de Segmentação + Modelo RMBG
            segmentador = await pipeline('image-segmentation', 'Xenova/bria-rmbg-1.4', {
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
            const resultado = await segmentador(dados.imagemUrl);
            
            // Código blindado: a IA pode devolver um Array ou um Objeto direto dependendo da imagem.
            // Aqui nós garantimos que vamos pegar os pixels da imagem recortada ("mask")
            const item = Array.isArray(resultado) ? resultado[0] : resultado;
            const imagemProcessada = item.mask || item;
            
            // Convertendo os dados brutos para uma imagem PNG transparente
            const canvas = new OffscreenCanvas(imagemProcessada.width, imagemProcessada.height);
            const ctx = canvas.getContext('2d');
            
            const imgData = new ImageData(
                new Uint8ClampedArray(imagemProcessada.data), 
                imagemProcessada.width, 
                imagemProcessada.height
            );
            ctx.putImageData(imgData, 0, 0);
            
            const blob = await canvas.convertToBlob({ type: 'image/png' });
            
            self.postMessage({ status: 'concluido', resultadoBlob: blob });
        } catch (erro) {
            self.postMessage({ status: 'erro', mensagem: erro.message });
        }
    }
});
