// worker.js
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers';

env.allowLocalModels = false;
let removedorFundo = null;

self.addEventListener('message', async (event) => {
    const dados = event.data;

    // 1. Comando para carregar o modelo de IA
    if (dados.acao === 'carregar') {
        self.postMessage({ status: 'iniciando' });
        
        try {
            // Usamos o pipeline exclusivo para remoção de fundo (novo na v3)
            removedorFundo = await pipeline('background-removal', 'briaai/RMBG-1.4', {
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
        if (!removedorFundo) return;
        
        self.postMessage({ status: 'processando' });
        
        try {
            // A IA remove o fundo e devolve a imagem com transparência
            const resultado = await removedorFundo(dados.imagemUrl);
            
            // Converte os pixels para uma imagem PNG real
            const canvas = new OffscreenCanvas(resultado.width, resultado.height);
            const ctx = canvas.getContext('2d');
            
            const imgData = new ImageData(
                new Uint8ClampedArray(resultado.data), 
                resultado.width, 
                resultado.height
            );
            ctx.putImageData(imgData, 0, 0);
            
            const blob = await canvas.convertToBlob({ type: 'image/png' });
            
            self.postMessage({ status: 'concluido', resultadoBlob: blob });
        } catch (erro) {
            self.postMessage({ status: 'erro', mensagem: erro.message });
        }
    }
});
