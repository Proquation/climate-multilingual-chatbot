Here's the enhanced version of your bakeoff script with all the requested features:

```python
import json
import os
import time
import pandas as pd
import evaluate
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# --- Configuration ---
# The four models to be tested head-to-head.
# The script will automatically skip any that can't be accessed.
MODELS_TO_TEST = {
    "Nova Pro": "amazon.nova-pro-v1:0",
    "Nova Micro": "amazon.nova-micro-v1:0",
    "Nova Lite": "amazon.nova-lite-v1:0",
    "Titan Text Premier": "amazon.titan-text-premier-v1:0"
}

# Approximate cost per 1K tokens (you should verify these with AWS pricing)
MODEL_COSTS = {
    "amazon.nova-pro-v1:0": {"input": 0.0008, "output": 0.0032},
    "amazon.nova-micro-v1:0": {"input": 0.000075, "output": 0.0003},
    "amazon.nova-lite-v1:0": {"input": 0.00006, "output": 0.00024},
    "amazon.titan-text-premier-v1:0": {"input": 0.0005, "output": 0.0015}
}

LANGUAGES = {
    "Arabic": "ar",
    "Chinese (Simplified)": "zh",
    "Russian": "ru",
    "Japanese": "ja",
    "Hindi": "hi",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Filipino (Tagalog)": "tl"
}

TEST_PHRASES = [
    "Climate change is a long-term shift in global weather patterns and temperatures.",
    "What are the most effective strategies for mitigating the impacts of climate change in coastal communities?",
    "Renewable energy sources, such as solar and wind power, are crucial for reducing carbon emissions.",
    "The agricultural sector must adapt to changing climate conditions to ensure food security.",
    "International cooperation is essential for addressing the global challenge of climate change."
]

# In a real-world scenario, these would be high-quality, human-generated translations.
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
        "जलवायु परिवर्तन की वैश्विक चुनौती से निपटने के लिए अंतर्राष्ट्रीय सहयोग आवश्यक है।"
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
    ],
    "tl": [
        "Ang pagbabago ng klima ay isang pangmatagalang pagbabago sa mga pattern ng panahon at temperatura sa buong mundo.",
        "Ano ang mga pinaka-epektibong estratehiya para sa pagpapagaan ng mga epekto ng pagbabago ng klima sa mga komunidad sa baybayin?",
        "Ang mga renewable na pinagkukunan ng enerhiya, tulad ng solar at wind power, ay mahalaga para sa pagbawas ng carbon emissions.",
        "Ang sektor ng agrikultura ay dapat mag-adapt sa mga nagbabagong kondisyon ng klima upang matiyak ang seguridad sa pagkain.",
        "Ang internasyonal na kooperasyon ay mahalaga para sa pagharap sa pandaigdigang hamon ng pagbabago ng klima."
    ]
}

# Cache configuration
CACHE_FILE = 'translation_cache.json'

# Initialize the Boto3 client for Bedrock Runtime
try:
    bedrock_client = boto3.client('bedrock-runtime')
except Exception as e:
    print(f"Error creating Boto3 client: {e}")
    print("Please ensure your AWS credentials and region are configured correctly.")
    exit()

def load_cache():
    """Load cached translations if available."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """Save cache to file."""
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)

def estimate_tokens(text):
    """Rough estimation of tokens (4 chars per token approximation)."""
    return len(text) / 4

def calculate_cost(model_id, input_text, output_text):
    """Calculate estimated cost based on token usage."""
    if model_id not in MODEL_COSTS:
        return 0
    
    input_tokens = estimate_tokens(input_text) / 1000  # Convert to K tokens
    output_tokens = estimate_tokens(output_text) / 1000  # Convert to K tokens
    
    cost = (input_tokens * MODEL_COSTS[model_id]["input"] + 
            output_tokens * MODEL_COSTS[model_id]["output"])
    
    return cost

def translate_text(model_name, model_id, text, target_lang_name, cache):
    """
    Translates text using a specified Bedrock model via Boto3,
    with retries for throttling and graceful failure for access errors.
    """
    # Check cache first
    cache_key = f"{model_id}_{target_lang_name}_{text}"
    if cache_key in cache:
        cached_result = cache[cache_key]
        return cached_result['translation'], cached_result['response_time'], True
    
    prompt = f"""
    You are a professional translator.
    Translate the following English text to {target_lang_name}.
    Style: Formal
    Tone: Informative
    Glossary:
        - "Climate change" should be translated consistently.
        - "Carbon emissions" should be translated consistently.

    English text to translate: "{text}"
    Translation:
    """
    request_body = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 1024,
            "temperature": 0.1,
            "topP": 0.9,
            "stopSequences": []
        }
    }

    max_retries = 3
    start_time = time.time()
    
    for attempt in range(max_retries):
        try:
            response = bedrock_client.invoke_model(
                modelId=model_id,
                contentType='application/json',
                accept='application/json',
                body=json.dumps(request_body)
            )
            response_body = json.loads(response['body'].read())
            translation = response_body['results'][0]['outputText'].strip()
            response_time = time.time() - start_time
            
            # Cache the result
            cache[cache_key] = {
                'translation': translation,
                'response_time': response_time
            }
            
            return translation, response_time, False
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                print(f"  ...Rate limit hit for {model_name}. Retrying in {2**attempt}s...")
                time.sleep(2**attempt)
                continue
            elif error_code in ['AccessDeniedException', 'ValidationException']:
                print(f"  ...SKIPPING {model_name} ({model_id}). Reason: {error_code}.")
                return None, 0, False
            else:
                print(f"  ...An unhandled error occurred for {model_name}: {e}")
                return None, 0, False
    print(f"  ...Failed to get a response from {model_name} after {max_retries} retries.")
    return None, 0, False

def create_visualizations(results_df):
    """Create charts and visualizations for the report."""
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Quality Scores Heatmap
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # BLEU scores heatmap
    bleu_pivot = results_df.pivot_table(values='BLEU', index='Model', columns='Language', aggfunc='mean')
    sns.heatmap(bleu_pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax1, vmin=0, vmax=1)
    ax1.set_