import re

# 1. Simulate a knowledge base (domain-specific documents)
# This represents the external, up-to-date, or proprietary data
# that an LLM might not have been trained on.
KNOWLEDGE_BASE = [
    "Türkiye'nin başkenti Ankara'dır.",
    "İstanbul, Türkiye'nin en kalabalık şehridir ve Boğaz Köprüsü'ne ev sahipliği yapar.",
    "Ayasofya, İstanbul'da bulunan tarihi bir yapıdır.",
    "RAG (Retrieval-Augmented Generation), büyük dil modellerine dış kaynaklardan bilgi ekleyerek yanıt üretme yöntemidir.",
    "Fine-tuning, mevcut bir büyük dil modelini belirli bir veri kümesi üzerinde yeniden eğiterek özelleştirme işlemidir.",
    "Halüsinasyon, LLM'lerin yanlış veya uydurma bilgiler üretmesidir.",
    "Bu makale RAG ve Fine-tuning stratejilerini karşılaştırmaktadır."
]

def retrieve_documents(query: str, knowledge_base: list[str], top_k: int = 2) -> list[str]:
    """
    Simulates retrieval of relevant documents from a knowledge base based on a query.
    In a real RAG system, this would involve embeddings and vector similarity search.
    Here, we use a simple keyword matching for demonstration purposes.
    """
    query_words = set(query.lower().split())
    
    scored_documents = []
    for doc in knowledge_base:
        # Use regex to find words, then convert to set for intersection
        doc_words = set(re.findall(r'\b\w+\b', doc.lower())) 
        overlap = len(query_words.intersection(doc_words))
        if overlap > 0:
            scored_documents.append((overlap, doc))
            
    scored_documents.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_documents[:top_k]]

def simulate_llm_response(original_query: str, context: list[str]) -> str:
    """
    Simulates an LLM generating a response using the provided context.
    In a real scenario, this would be an API call to an LLM with a carefully constructed prompt
    that includes the retrieved context. Here, we demonstrate how the context influences the response.
    """
    if context:
        # --- RAG Adımı 2: Üretimi Destekleme (Augmentation) ve Yanıt Üretme (Generation) ---
        # Geri çağrılan bağlamı kullanarak LLM'in yanıtını şekillendiririz.
        # Bu, LLM'in doğru ve güncel bilgiye dayanmasını sağlar, halüsinasyonu azaltır.
        
        context_str = "\n".join([f"- {c}" for c in context])
        
        response_parts = []
        response_parts.append(f"LLM (RAG ile, bağlam kullanıldı):")
        response_parts.append(f"  Soru: '{original_query}'")
        response_parts.append(f"  Kullanılan Bağlam:")
        for line in context_str.split('\n'):
            response_parts.append(f"    {line}")
        
        # A very basic attempt to "answer" using keywords from context for demonstration
        found_answer = False
        for doc in context:
            if "başkent Ankara" in doc.lower() and "başkent" in original_query.lower():
                response_parts.append("  Cevap: Türkiye'nin başkenti Ankara'dır.")
                found_answer = True
                break
            elif "istanbul" in doc.lower() and "kalabalık" in original_query.lower():
                response_parts.append("  Cevap: Türkiye'nin en kalabalık şehri İstanbul'dur.")
                found_answer = True
                break
            elif "rag" in doc.lower() and "nedir" in original_query.lower():
                response_parts.append("  Cevap: RAG (Retrieval-Augmented Generation), büyük dil modellerine dış kaynaklardan bilgi ekleyerek yanıt üretme yöntemidir.")
                found_answer = True
                break
            elif "fine-tuning" in doc.lower() and "nedir" in original_query.lower():
                response_parts.append("  Cevap: Fine-tuning, mevcut bir büyük dil modelini belirli bir veri kümesi üzerinde yeniden eğiterek özelleştirme işlemidir.")
                found_answer = True
                break
            elif "halüsinasyon" in doc.lower() and "nedir" in original_query.lower():
                response_parts.append("  Cevap: Halüsinasyon, LLM'lerin yanlış veya uydurma bilgiler üretmesidir.")
                found_answer = True
                break
            elif "makale" in doc.lower() and ("konu" in original_query.lower() or "ne hakkında" in original_query.lower()):
                response_parts.append("  Cevap: Bu makale RAG ve Fine-tuning stratejilerini karşılaştırmaktadır.")
                found_answer = True
                break
        
        if not found_answer:
            response_parts.append("  Cevap: Verilen bağlamda doğrudan bir yanıt bulunamadı, ancak ilgili bilgiler sunuldu.")
            
        return "\n".join(response_parts)
    else:
        # Simulate an LLM without specific context (might hallucinate or be generic).
        # This demonstrates the problem RAG aims to solve by providing up-to-date/specific info.
        return (
            f"LLM (bağlam olmadan):\n"
            f"  Soru: '{original_query}'\n"
            f"  Cevap: Üzgünüm, bu konuda özel bir bilgiye sahip değilim. "
            f"Belki de bir halüsinasyon riski var veya genel bilgim yeterli değil."
        )

def main():
    print("RAG (Retrieval-Augmented Generation) Simülasyonu\n")

    queries = [
        "Türkiye'nin başkenti neresidir?",
        "İstanbul'un özellikleri nelerdir?",
        "RAG nedir?",
        "Fine-tuning ne anlama gelir?",
        "Halüsinasyon nedir?",
        "Bu makale ne hakkında?",
        "Mars'ta yaşam var mı?" # Query for which knowledge base has no info, demonstrating RAG's limitation without relevant data
    ]

    for i, query in enumerate(queries):
        print(f"---\nSoru {i+1}: {query}\n---")
        
        # --- RAG Adımı 1: Bilgi Geri Çağırma (Retrieval) ---
        # Makalede bahsedilen "Geri Çağırma Destekli Üretim"in ilk adımı.
        # Sorguya en uygun bilgileri 'bilgi tabanından' buluruz.
        retrieved_context = retrieve_documents(query, KNOWLEDGE_BASE)
        
        print(f"Geri Çağrılan Bağlam ({len(retrieved_context)} belge):")
        if retrieved_context:
            for doc in retrieved_context:
                print(f"  - {doc}")
        else:
            print("  - Bağlam bulunamadı.")
            
        llm_response = simulate_llm_response(query, retrieved_context)
        print(f"{llm_response}\n")

if __name__ == "__main__":
    main()
