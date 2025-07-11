"""
Pipeline principal para previsão do IBOVESPA usando regressão
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_utils import carregar_dados_ibovespa, preparar_dataset_temporal
from feature_engineering import criar_features_basicas, criar_features_avancadas, criar_targets
from model_utils import criar_modelo_otimizado, avaliar_modelo
from correlation_analysis import analisar_correlacoes, selecionar_features_otimizadas

def pipeline_completo():
    """
    Pipeline completo para previsão do IBOVESPA
    """
    print("="*70)
    print("🤖 PIPELINE ML - PREVISÃO IBOVESPA")
    print("📊 ABORDAGEM: Regressão + Análise de Correlação")
    print("🎯 META: 75% de acurácia nos últimos 30 dias")
    print("="*70)
    
    try:
        # 1. CARREGAMENTO DE DADOS
        print("\n📥 ETAPA 1: Carregamento de Dados")
        data = carregar_dados_ibovespa(anos=10)
        
        # 2. ENGENHARIA DE FEATURES
        print("\n🔧 ETAPA 2: Engenharia de Features")
        data, features_basicas = criar_features_basicas(data)
        data, features_completas = criar_features_avancadas(data, features_basicas)
        
        # 3. CRIAÇÃO DE TARGETS
        print("\n🎯 ETAPA 3: Criação de Targets")
        data = criar_targets(data)
        
        # 4. ANÁLISE DE CORRELAÇÃO
        print("\n🔍 ETAPA 4: Análise de Correlação")
        analise = analisar_correlacoes(data, features_completas)
        
        # 5. SELEÇÃO DE FEATURES
        print("\n⚡ ETAPA 5: Seleção Otimizada de Features")
        features_selecionadas = selecionar_features_otimizadas(analise, features_completas)
        
        # 6. PREPARAÇÃO DO DATASET
        print("\n📊 ETAPA 6: Preparação do Dataset")
        X_treino, X_teste, y_treino, y_teste, y_treino_dir, y_teste_dir = preparar_dataset_temporal(
            data, features_selecionadas, n_test_days=30
        )
        
        # 7. CRIAÇÃO E TREINAMENTO DO MODELO
        print("\n🤖 ETAPA 7: Criação do Modelo Otimizado")
        modelo = criar_modelo_otimizado()
        
        # 8. AVALIAÇÃO
        print("\n📈 ETAPA 8: Avaliação Final")
        resultados = avaliar_modelo(
            modelo, X_treino, X_teste, y_treino, y_teste, y_treino_dir, y_teste_dir
        )
        
        # 9. RELATÓRIO FINAL
        print("\n" + "="*70)
        print("📋 RELATÓRIO FINAL")
        print("="*70)
        
        acuracia = resultados['acuracia_teste']
        print(f"🎯 META: 75% de acurácia")
        print(f"📊 RESULTADO: {acuracia:.1%}")
        print(f"🏆 STATUS: {'✅ SUCESSO' if acuracia >= 0.75 else '❌ NÃO ATINGIDA'}")
        print(f"🔧 FEATURES: {len(features_selecionadas)} selecionadas")
        print(f"📈 R²: {resultados['r2_teste']:.4f}")
        
        # Baseline
        baseline = max(y_teste_dir.mean(), 1 - y_teste_dir.mean())
        melhoria = acuracia - baseline
        print(f"\n📊 COMPARAÇÃO:")
        print(f"   Baseline: {baseline:.1%}")
        print(f"   Modelo: {acuracia:.1%}")
        print(f"   Melhoria: {melhoria*100:+.1f} pontos")
        
        if acuracia >= 0.75:
            print("\n🎉 MISSÃO CUMPRIDA!")
            print("   Meta de 75% de acurácia atingida!")
        else:
            print(f"\n📊 Meta não atingida")
            print(f"   Faltam {(0.75 - acuracia)*100:.1f} pontos percentuais")
            print("\n💡 Sugestões:")
            print("   - Mais dados macroeconômicos")
            print("   - Modelos não-lineares")
            print("   - Dados intraday")
        
        print("\n✅ PIPELINE CONCLUÍDO!")
        
        return {
            'acuracia': acuracia,
            'meta_atingida': acuracia >= 0.75,
            'features_usadas': features_selecionadas,
            'resultados_completos': resultados
        }
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    resultado = pipeline_completo()
    
    if resultado:
        if resultado['meta_atingida']:
            print(f"\n🏆 SUCESSO! Acurácia: {resultado['acuracia']:.1%}")
        else:
            print(f"\n📊 Acurácia alcançada: {resultado['acuracia']:.1%}")
    else:
        print("\n❌ Pipeline falhou")
