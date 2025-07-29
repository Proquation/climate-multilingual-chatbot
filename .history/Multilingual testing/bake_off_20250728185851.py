import asyncio
import pandas as pd
import evaluate
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.nova_flow import BedrockModel

# --- Configuration ---
MODELS_TO_TEST = {
    "Micro Nova": "amazon.nova-micro-v1:0",
    "Nova Pro": "amazon.nova-pro-v1:0",
    "Nova Premier": "amazon.nova-premier-v1:0"
}

LANGUAGES = {
    "Arabic": "ar",
    "Chinese (Simplified)": "zh",
    "Russian": "ru",
    "Japanese": "ja",
    "Hindi": "hi",
    "German": "de",
    "French": "fr",
    "Spanish": "es"
}

TEST_PHRASES = [
    "Climate change is a long-term shift in global weather patterns and temperatures.",
    "What are the most effective strategies for mitigating the impacts of climate change in coastal communities?",
    "Renewable energy sources, such as solar and wind power, are crucial for reducing carbon emissions.",
    "The agricultural sector must adapt to changing climate conditions to ensure food security.",
    "International cooperation is essential for addressing the global challenge of climate change."
]

# --- Reference Translations (for evaluation) ---
# In a real-world scenario, these would be high-quality, human-generated translations.
# For this bake-off, we will use a placeholder.
# NOTE: Using a model to generate references and then evaluating against them is not a robust methodology.
REFERENCE_TRANSLATIONS = {
    "ar": [
        "تغير المناخ هو تحول طويل الأمد في أنماط الطقس ودرجات الحرارة العالمية.",
        "ما هي أكثر الاستراتيجيات فعالية للتخفيف من آثار تغير المناخ في المجتمعات الساحلية؟",
        "تعد مصادر الطاقة المتجددة، مثل الطاقة الشمسية وطاقة الرياح، حاسمة لخفض انبعاثات الكربون.",
        "يجب على القطاع الزراعي التكيف مع الظروف المناخية المتغيرة لضمان الأمن الغذائي.",
        "التعاون الدولي ضروري لمواجهة التحدي العالمي المتمثل في تغير المناخ."
    ],
    "zh": [
        "气候变化是全球天气模式和温度的长期转变。",
        "在沿海社区，减缓气候变化影响最有效的策略是什么？",
        "太阳能和风能等可再生能源对于减少碳排放至关重要。",
        "农业部门必须适应不断变化的气候条件，以确保粮食安全。",
        "国际合作对于应对气候变化的全球挑战至关重要。"
    ],
    "ru": [
        "Изменение климата - это долгосрочное изменение глобальных погодных условий и температур.",
        "Каковы наиболее эффективные стратегии смягчения последствий изменения климата в прибрежных сообществах?",
        "Возобновляемые источники энергии, такие как солнечная и ветровая энергия, имеют решающее значение для сокращения выбросов углерода.",
        "Сельскохозяйственный сектор должен адаптироваться к изменяющимся климатическим условиям для обеспечения продовольственной безопасности.",
        "Международное сотрудничество необходимо для решения глобальной проблемы изменения климата."
    ],
    "ja": [
        "気候変動とは、地球規模の気象パターンと気温の長期的な変化です。",
        "沿岸地域社会における気候変動の影響を緩和するための最も効果的な戦略は何ですか？",
        "太陽光や風力などの再生可能エネルギー源は、炭素排出量を削減するために不可欠です。",
        "食料安全保障を確保するためには、農業部門は変化する気候条件に適応する必要があります。",
        "気候変動という世界的な課題に取り組むためには、国際協力が不可欠です。"
    ],
    "hi": [
        "जलवायु परिवर्तन वैश्विक मौसम पैटर्न और तापमान में एक दीर्घकालिक बदलाव है।",
        "तटीय समुदायों में जलवायु परिवर्तन के प्रभावों को कम करने के लिए सबसे प्रभावी रणनीतियाँ क्या हैं?",
        "सौर और पवन ऊर्जा जैसे नवीकरणीय ऊर्जा स्रोत कार्बन उत्सर्जन को कम करने के लिए महत्वपूर्ण हैं।",
        "खाद्य सुरक्षा सुनिश्चित करने के लिए कृषि क्षेत्र को बदलती जलवायु परिस्थितियों के अनुकूल होना चाहिए।",
        "जलवायु परिवर्तन की वैश्विक चुनौती से निपटने के लिए अंतर्राष्ट्रीय सहयोग आवश्यक है。"
    ],
    "de": [
        "Der Klimawandel ist eine langfristige Veränderung der globalen Wettermuster und Temperaturen.",
        "Was sind die wirksamsten Strategien zur Minderung der Auswirkungen des Klimawandels in Küstengemeinden?",
        "Erneuerbare Energiequellen wie Sonnen- und Windenergie sind entscheidend für die Reduzierung der Kohlenstoffemissionen.",
        "Der Agrarsektor muss sich an die veränderten klimatischen Bedingungen anpassen, um die Ernährungssicherheit zu gewährleisten.",
        "Internationale Zusammenarbeit ist unerlässlich, um die globale Herausforderung des Klimawandels zu bewältigen."
    ],
    "fr": [
        "Le changement climatique est une modification à long terme des régimes météorologiques et des températures mondiales.",
        "Quelles sont les stratégies les plus efficaces pour atténuer les impacts du changement climatique dans les communautés côtières ?",
        "Les sources d'énergie renouvelables, telles que l'énergie solaire et éolienne, sont cruciales pour réduire les émissions de carbone.",
        "Le secteur agricole doit s'adapter aux conditions climatiques changeantes pour assurer la sécurité alimentaire.",
        "La coopération internationale est essentielle pour relever le défi mondial du changement climatique."
    ],
    "es": [
        "El cambio climático es una alteración a largo plazo de los patrones meteorológicos y las temperaturas globales.",
        "¿Cuáles son las estrategias más eficaces para mitigar los impactos del cambio climático en las comunidades costeras?",
        "Las fuentes de energía renovables, como la solar y la eólica, son cruciales para reducir las emisiones de carbono.",
        "El sector agrícola debe adaptarse a las condiciones climáticas cambiantes para garantizar la seguridad alimentaria.",
        "La cooperación internacional es esencial para abordar el desafío global del cambio climático."
    ]
}


async def translate_text(model_name, model_id, text, target_lang_code, target_lang_name):
    """
    Translates text using a specified Bedrock model.
    """
    structured_prompt = f"""
    Translate the following English text to {target_lang_name}.
    Style: Formal
    Tone: Informative
    Glossary: 
        - "Climate change" should be translated consistently.
        - "Carbon emissions" should be translated consistently.

    Text to translate: "{text}"
    """
    try:
        model = BedrockModel(model_id=model_id)
        # The nova_translation function in the current implementation does not support passing a structured prompt.
        # We will call it directly with the text and target language.
        # The temperature is hardcoded to 0.1 in the nova_translation function.
        translation = await model.nova_translation(text, source_lang="en", target_lang=target_lang_code)
        return translation
    except Exception as e:
        print(f"Error translating with {model_name}: {e}")
        return None

async def main():
    """
    Main function to run the bake-off.
    """
    results = []
    bleu = evaluate.load('bleu')
    chrf = evaluate.load('chrf')
    # COMET is not available in the datasets library. We will skip it.

    for model_name, model_id in MODELS_TO_TEST.items():
        for lang_name, lang_code in LANGUAGES.items():
            for i, phrase in enumerate(TEST_PHRASES):
                print(f"Translating with {model_name} to {lang_name}: \"{phrase}\"")
                translation = await translate_text(model_name, model_id, phrase, lang_code, lang_name)

                if translation:
                    bleu_score = bleu.compute(predictions=[translation], references=[REFERENCE_TRANSLATIONS[lang_code][i]])['bleu']
                    chrf_score = chrf.compute(predictions=[translation], references=[REFERENCE_TRANSLATIONS[lang_code][i]])['score']

                    results.append({
                        "Model": model_name,
                        "Language": lang_name,
                        "Source Text": phrase,
                        "Translated Text": translation,
                        "Reference Text": REFERENCE_TRANSLATIONS[lang_code][i],
                        "BLEU": bleu_score,
                        "chrF": chrf_score
                    })

    results_df = pd.DataFrame(results)
    print("\n--- Bake-off Results ---")
    print(results_df)

    # --- Reporting ---
    print("\n--- Performance Report ---")
    summary = results_df.groupby('Model')[['BLEU', 'chrF']].mean()
    print(summary)

    winner = summary.mean(axis=1).idxmax()
    print(f"\n--- Winner ---")
    print(f"The winning model is: {winner}")

if __name__ == "__main__":
    asyncio.run(main())