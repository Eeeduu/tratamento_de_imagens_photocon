// worker.js
// Atualizado para a versão 3 oficial da Hugging Face
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers';

env.allowLocalModels = false;
let segmentador = null;

self.addEventListener('message', async (event) => {
    const dados = event.data;

    // 1. Comando para carregar o modelo de IA
    if (dados.acao === 'carregar') {
        self.postMessage({ status: 'iniciando' });
        
        try {
            // Usando a versão otimizada (ONNX) do modelo para não travar o navegador
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
            // A IA analisa a imagem
            const resultado = await segmentador(dados.imagemUrl);
            
            // Modelos de segmentação na V3 geralmente retornam um array.
            // Pegamos a máscara que contém a imagem já cortada.
            const imagemProcessada = Array.isArray(resultado) ? resultado[0].mask : resultado;
            
            // Transformamos os dados brutos da IA em uma imagem PNG usando OffscreenCanvas
            const canvas = new OffscreenCanvas(imagemProcessada.width, imagemProcessada.height);
            const ctx = canvas.getContext('2d');
            
            // Converte os pixels para um formato que o Canvas entenda
            const imgData = new ImageData(
                new Uint8ClampedArray(imagemProcessada.data), 
                imagemProcessada.width, 
                imagemProcessada.height
            );
            ctx.putImageData(imgData, 0, 0);
            
            // Gera o arquivo final da imagem transparente
            const blob = await canvas.convertToBlob({ type: 'image/png' });
            
            self.postMessage({ status: 'concluido', resultadoBlob: blob });
        } catch (erro) {
            self.postMessage({ status: 'erro', mensagem: erro.message });
        }
    }
});
