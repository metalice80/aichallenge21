# Оценка RAG-чата и памяти задачи

Дата: 2026-10-04T22:32:38.967689+00:00
Режим проверки: фиктивная локальная интеграция без Ollama, сети, embedding-модели и CrossEncoder
LLM: `deterministic-offline-fake`
Retrieval: `{"candidate_top_k": 5, "final_top_k": 3, "similarity_threshold": 0.35, "rerank_threshold": null, "use_query_rewrite": true, "use_similarity_filter": true, "reranker_type": "heuristic", "max_alternative_queries": 2, "max_chunks_per_source": 3}`

История, task state, conversation summary, recent messages и RAG-контекст проверяются как отдельные сущности.

## Настройка локального RAG для русскоязычных PDF

| Ход | Статус | Цель | Ограничения | Follow-up | Source hit | Источники | Выдумано | Время, с |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | completed | 1 | 1 | 1 | 0 | 1 | 0 | 0.005 |
| 2 | completed | 1 | 1 | 1 | 1 | 1 | 0 | 0.004 |
| 3 | completed | 1 | 1 | 1 | 1 | 1 | 0 | 0.003 |
| 4 | completed | 1 | 1 | 1 | 1 | 1 | 0 | 0.005 |
| 5 | completed | 1 | 1 | 1 | 1 | 1 | 0 | 0.005 |
| 6 | completed | 1 | 1 | 1 | 1 | 1 | 0 | 0.004 |
| 7 | completed | 1 | 1 | 1 | 0 | 1 | 0 | 0.004 |
| 8 | completed | 1 | 1 | 1 | 1 | 1 | 0 | 0.005 |
| 9 | completed | 1 | 1 | 1 | 1 | 1 | 0 | 0.005 |
| 10 | completed | 1 | 1 | 1 | 1 | 1 | 0 | 0.003 |
| 11 | completed | 1 | 1 | 1 | 1 | 1 | 0 | 0.003 |
| 12 | completed | 1 | 1 | 1 | 1 | 1 | 0 | 0.003 |

### Детали памяти и источников

#### Ход 1

- Сообщение: Моя цель — настроить локальный RAG для русскоязычных PDF. Работаем только на CPU, без внешних API, используем structural index, а Final Top-K должен быть не больше 3.
- Standalone query: Моя цель — настроить локальный RAG для русскоязычных PDF. Работаем только на CPU, без внешних API, используем structural index, а Final Top-K должен быть не больше 3.
- Ожидаемые источники: ['README.md']
- Использованные источники: ['day23_reranking_filtering_omp.md']
- Состояние памяти: `{"goal": "Настроить локальный RAG для русскоязычных PDF", "constraints": ["Только CPU", "Без внешних API", "Использовать structural index", "Final Top-K не больше 3"], "terms": {}, "confirmed_facts": [], "decisions": [], "open_questions": [], "user_preferences": [], "version": 1, "updated_at": "2026-10-04T22:32:38.763490+00:00"}`

#### Ход 2

- Сообщение: Под термином «основной индекс» дальше понимай structural index.
- Standalone query: Под термином «основной индекс» дальше понимай structural index.
- Ожидаемые источники: ['README.md']
- Использованные источники: ['README.md']
- Состояние памяти: `{"goal": "Настроить локальный RAG для русскоязычных PDF", "constraints": ["Только CPU", "Без внешних API", "Использовать structural index", "Final Top-K не больше 3"], "terms": {"основной индекс": "structural index"}, "confirmed_facts": [], "decisions": [], "open_questions": [], "user_preferences": [], "version": 2, "updated_at": "2026-10-04T22:32:38.773742+00:00"}`

#### Ход 3

- Сообщение: Как его построить для загруженных PDF?
- Standalone query: Как построить structural index для загруженных PDF?
- Ожидаемые источники: ['README.md', 'guide.md']
- Использованные источники: ['README.md']
- Состояние памяти: `{"goal": "Настроить локальный RAG для русскоязычных PDF", "constraints": ["Только CPU", "Без внешних API", "Использовать structural index", "Final Top-K не больше 3"], "terms": {"основной индекс": "structural index"}, "confirmed_facts": [], "decisions": [], "open_questions": [], "user_preferences": [], "version": 2, "updated_at": "2026-10-04T22:32:38.773742+00:00"}`

#### Ход 4

- Сообщение: Измени ограничение Final Top-K: теперь допускается не больше 4 вместо 3.
- Standalone query: Измени ограничение Final Top-K: теперь допускается не больше 4 вместо 3.
- Ожидаемые источники: ['day23_reranking_filtering_omp.md']
- Использованные источники: ['day23_reranking_filtering_omp.md']
- Состояние памяти: `{"goal": "Настроить локальный RAG для русскоязычных PDF", "constraints": ["Только CPU", "Без внешних API", "Использовать structural index", "Final Top-K не больше 4"], "terms": {"основной индекс": "structural index"}, "confirmed_facts": [], "decisions": [], "open_questions": [], "user_preferences": [], "version": 3, "updated_at": "2026-10-04T22:32:38.792587+00:00"}`

#### Ход 5

- Сообщение: Ограничение «без внешних API» явно отменяю, но CPU оставляем.
- Standalone query: Ограничение «без внешних API» явно отменяю, но CPU оставляем.
- Ожидаемые источники: ['README.md']
- Использованные источники: ['README.md']
- Состояние памяти: `{"goal": "Настроить локальный RAG для русскоязычных PDF", "constraints": ["Только CPU", "Использовать structural index", "Final Top-K не больше 4"], "terms": {"основной индекс": "structural index"}, "confirmed_facts": [], "decisions": [], "open_questions": [], "user_preferences": [], "version": 4, "updated_at": "2026-10-04T22:32:38.801916+00:00"}`

#### Ход 6

- Сообщение: Побочный вопрос: требуется ли OCR для сканированных PDF?
- Standalone query: Побочный вопрос: требуется ли OCR для сканированных PDF?
- Ожидаемые источники: ['README.md']
- Использованные источники: ['README.md']
- Состояние памяти: `{"goal": "Настроить локальный RAG для русскоязычных PDF", "constraints": ["Только CPU", "Использовать structural index", "Final Top-K не больше 4"], "terms": {"основной индекс": "structural index"}, "confirmed_facts": [], "decisions": [], "open_questions": [], "user_preferences": [], "version": 4, "updated_at": "2026-10-04T22:32:38.801916+00:00"}`

#### Ход 7

- Сообщение: Вернёмся к основной цели: какие этапы проходят русскоязычные PDF до поиска?
- Standalone query: Вернёмся к основной цели: какие этапы проходят русскоязычные PDF до поиска?
- Ожидаемые источники: ['document_indexer_omp.md']
- Использованные источники: ['README.md']
- Состояние памяти: `{"goal": "Настроить локальный RAG для русскоязычных PDF", "constraints": ["Только CPU", "Использовать structural index", "Final Top-K не больше 4"], "terms": {"основной индекс": "structural index"}, "confirmed_facts": [], "decisions": [], "open_questions": [], "user_preferences": [], "version": 4, "updated_at": "2026-10-04T22:32:38.801916+00:00"}`

#### Ход 8

- Сообщение: Подтверждённый факт: все мои PDF содержат текстовый слой.
- Standalone query: Подтверждённый факт: все мои PDF содержат текстовый слой.
- Ожидаемые источники: нет
- Использованные источники: нет
- Состояние памяти: `{"goal": "Настроить локальный RAG для русскоязычных PDF", "constraints": ["Только CPU", "Использовать structural index", "Final Top-K не больше 4"], "terms": {"основной индекс": "structural index"}, "confirmed_facts": ["Все PDF содержат текстовый слой"], "decisions": [], "open_questions": [], "user_preferences": [], "version": 5, "updated_at": "2026-10-04T22:32:38.829073+00:00"}`

#### Ход 9

- Сообщение: Предпочитаю краткие ответы со списком команд.
- Standalone query: Предпочитаю краткие ответы со списком команд.
- Ожидаемые источники: ['README.md']
- Использованные источники: ['README.md']
- Состояние памяти: `{"goal": "Настроить локальный RAG для русскоязычных PDF", "constraints": ["Только CPU", "Использовать structural index", "Final Top-K не больше 4"], "terms": {"основной индекс": "structural index"}, "confirmed_facts": ["Все PDF содержат текстовый слой"], "decisions": [], "open_questions": [], "user_preferences": ["Краткие ответы со списком команд"], "version": 6, "updated_at": "2026-10-04T22:32:38.838415+00:00"}`

#### Ход 10

- Сообщение: Какая версия Kubernetes обязательна для этого индексатора?
- Standalone query: Какая версия Kubernetes обязательна для этого индексатора?
- Ожидаемые источники: нет
- Использованные источники: нет
- Состояние памяти: `{"goal": "Настроить локальный RAG для русскоязычных PDF", "constraints": ["Только CPU", "Использовать structural index", "Final Top-K не больше 4"], "terms": {"основной индекс": "structural index"}, "confirmed_facts": ["Все PDF содержат текстовый слой"], "decisions": [], "open_questions": [], "user_preferences": ["Краткие ответы со списком команд"], "version": 6, "updated_at": "2026-10-04T22:32:38.838415+00:00"}`

#### Ход 11

- Сообщение: Проверь, что отменённое ограничение про внешние API не вернулось, а лимит Top-K равен 4.
- Standalone query: Проверь, что отменённое ограничение про внешние API не вернулось, а лимит Top-K равен 4.
- Ожидаемые источники: нет
- Использованные источники: нет
- Состояние памяти: `{"goal": "Настроить локальный RAG для русскоязычных PDF", "constraints": ["Только CPU", "Использовать structural index", "Final Top-K не больше 4"], "terms": {"основной индекс": "structural index"}, "confirmed_facts": ["Все PDF содержат текстовый слой"], "decisions": [], "open_questions": [], "user_preferences": ["Краткие ответы со списком команд"], "version": 6, "updated_at": "2026-10-04T22:32:38.838415+00:00"}`

#### Ход 12

- Сообщение: Суммируй итоговый план настройки локального RAG для PDF с учётом цели и активных ограничений.
- Standalone query: Суммируй итоговый план настройки локального RAG для PDF с учётом цели и активных ограничений.
- Ожидаемые источники: ['README.md', 'document_indexer_omp.md']
- Использованные источники: ['README.md']
- Состояние памяти: `{"goal": "Настроить локальный RAG для русскоязычных PDF", "constraints": ["Только CPU", "Использовать structural index", "Final Top-K не больше 4"], "terms": {"основной индекс": "structural index"}, "confirmed_facts": ["Все PDF содержат текстовый слой"], "decisions": [], "open_questions": [], "user_preferences": ["Краткие ответы со списком команд"], "version": 6, "updated_at": "2026-10-04T22:32:38.838415+00:00"}`

## Повышение качества retrieval и Source Hit@3

| Ход | Статус | Цель | Ограничения | Follow-up | Source hit | Источники | Выдумано | Время, с |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | completed | 1 | 1 | 1 | 1 | 1 | 0 | 0.003 |
| 2 | completed | 1 | 1 | 1 | 1 | 1 | 0 | 0.003 |
| 3 | completed | 1 | 1 | 1 | 1 | 1 | 0 | 0.003 |
| 4 | completed | 1 | 1 | 1 | 1 | 1 | 0 | 0.003 |
| 5 | completed | 1 | 1 | 1 | 1 | 1 | 0 | 0.005 |
| 6 | completed | 1 | 1 | 1 | 1 | 1 | 0 | 0.004 |
| 7 | completed | 1 | 1 | 1 | 1 | 1 | 0 | 0.004 |
| 8 | completed | 1 | 1 | 0 | 1 | 1 | 0 | 0.004 |
| 9 | completed | 1 | 1 | 1 | 1 | 1 | 0 | 0.004 |
| 10 | completed | 1 | 1 | 1 | 1 | 1 | 0 | 0.005 |
| 11 | completed | 1 | 1 | 1 | 1 | 1 | 0 | 0.004 |
| 12 | completed | 1 | 1 | 1 | 1 | 1 | 0 | 0.004 |

### Детали памяти и источников

#### Ход 1

- Сообщение: Цель — повысить Source Hit@3. Нельзя менять embedding-модель, используем heuristic reranker, сохраняем разумное время ответа и сравниваем Baseline с Enhanced RAG.
- Standalone query: Цель — повысить Source Hit@3. Нельзя менять embedding-модель, используем heuristic reranker, сохраняем разумное время ответа и сравниваем Baseline с Enhanced RAG.
- Ожидаемые источники: ['day23_reranking_filtering_omp.md']
- Использованные источники: ['day23_reranking_filtering_omp.md']
- Состояние памяти: `{"goal": "Повысить Source Hit@3", "constraints": ["Не менять embedding-модель", "Использовать heuristic reranker", "Сохранить разумное время ответа", "Сравнивать Baseline и Enhanced RAG"], "terms": {}, "confirmed_facts": [], "decisions": [], "open_questions": [], "user_preferences": [], "version": 1, "updated_at": "2026-10-04T22:32:38.871168+00:00"}`

#### Ход 2

- Сообщение: Термин «качество» в этом диалоге означает Source Hit@3, а не качество генерации.
- Standalone query: Термин «качество» в этом диалоге означает Source Hit@3, а не качество генерации.
- Ожидаемые источники: ['day23_reranking_filtering_omp.md']
- Использованные источники: ['day23_reranking_filtering_omp.md']
- Состояние памяти: `{"goal": "Повысить Source Hit@3", "constraints": ["Не менять embedding-модель", "Использовать heuristic reranker", "Сохранить разумное время ответа", "Сравнивать Baseline и Enhanced RAG"], "terms": {"качество": "Source Hit@3 retrieval"}, "confirmed_facts": [], "decisions": [], "open_questions": [], "user_preferences": [], "version": 2, "updated_at": "2026-10-04T22:32:38.879610+00:00"}`

#### Ход 3

- Сообщение: С каких значений similarity threshold начать эксперимент?
- Standalone query: С каких значений similarity threshold начать эксперимент?
- Ожидаемые источники: ['day23_reranking_filtering_omp.md']
- Использованные источники: ['day23_reranking_filtering_omp.md']
- Состояние памяти: `{"goal": "Повысить Source Hit@3", "constraints": ["Не менять embedding-модель", "Использовать heuristic reranker", "Сохранить разумное время ответа", "Сравнивать Baseline и Enhanced RAG"], "terms": {"качество": "Source Hit@3 retrieval"}, "confirmed_facts": [], "decisions": [], "open_questions": [], "user_preferences": [], "version": 2, "updated_at": "2026-10-04T22:32:38.879610+00:00"}`

#### Ход 4

- Сообщение: А как он влияет на пустые результаты?
- Standalone query: Как similarity threshold влияет на пустые результаты?
- Ожидаемые источники: ['day23_reranking_filtering_omp.md']
- Использованные источники: ['day23_reranking_filtering_omp.md']
- Состояние памяти: `{"goal": "Повысить Source Hit@3", "constraints": ["Не менять embedding-модель", "Использовать heuristic reranker", "Сохранить разумное время ответа", "Сравнивать Baseline и Enhanced RAG"], "terms": {"качество": "Source Hit@3 retrieval"}, "confirmed_facts": [], "decisions": [], "open_questions": [], "user_preferences": [], "version": 2, "updated_at": "2026-10-04T22:32:38.879610+00:00"}`

#### Ход 5

- Сообщение: Решение: начнём с threshold 0.35 и сравним с 0.30 и 0.40.
- Standalone query: Решение: начнём с threshold 0.35 и сравним с 0.30 и 0.40.
- Ожидаемые источники: ['day23_reranking_filtering_omp.md']
- Использованные источники: ['day23_reranking_filtering_omp.md']
- Состояние памяти: `{"goal": "Повысить Source Hit@3", "constraints": ["Не менять embedding-модель", "Использовать heuristic reranker", "Сохранить разумное время ответа", "Сравнивать Baseline и Enhanced RAG"], "terms": {"качество": "Source Hit@3 retrieval"}, "confirmed_facts": [], "decisions": ["Сравнить threshold 0.30, 0.35 и 0.40"], "open_questions": [], "user_preferences": [], "version": 3, "updated_at": "2026-10-04T22:32:38.901748+00:00"}`

#### Ход 6

- Сообщение: Уменьши Final Top-K с 5 до 3 и считай 3 новым активным значением.
- Standalone query: Уменьши Final Top-K с 5 до 3 и считай 3 новым активным значением.
- Ожидаемые источники: ['day23_reranking_filtering_omp.md']
- Использованные источники: ['day23_reranking_filtering_omp.md']
- Состояние памяти: `{"goal": "Повысить Source Hit@3", "constraints": ["Не менять embedding-модель", "Использовать heuristic reranker", "Сохранить разумное время ответа", "Сравнивать Baseline и Enhanced RAG", "Final Top-K 3"], "terms": {"качество": "Source Hit@3 retrieval"}, "confirmed_facts": [], "decisions": ["Сравнить threshold 0.30, 0.35 и 0.40"], "open_questions": [], "user_preferences": [], "version": 4, "updated_at": "2026-10-04T22:32:38.911078+00:00"}`

#### Ход 7

- Сообщение: Нужен ли Query Rewrite для неоднозначных вопросов?
- Standalone query: Нужен ли Query Rewrite для неоднозначных вопросов?
- Ожидаемые источники: ['day23_reranking_filtering_omp.md']
- Использованные источники: ['day23_reranking_filtering_omp.md']
- Состояние памяти: `{"goal": "Повысить Source Hit@3", "constraints": ["Не менять embedding-модель", "Использовать heuristic reranker", "Сохранить разумное время ответа", "Сравнивать Baseline и Enhanced RAG", "Final Top-K 3"], "terms": {"качество": "Source Hit@3 retrieval"}, "confirmed_facts": [], "decisions": ["Сравнить threshold 0.30, 0.35 и 0.40"], "open_questions": [], "user_preferences": [], "version": 4, "updated_at": "2026-10-04T22:32:38.911078+00:00"}`

#### Ход 8

- Сообщение: Используй его для следующего запроса: где он сохраняет диагностику?
- Standalone query: Как Query Rewrite помогает найти, где сохраняется диагностика?
- Ожидаемые источники: ['day23_reranking_filtering_omp.md']
- Использованные источники: ['day23_reranking_filtering_omp.md']
- Состояние памяти: `{"goal": "Повысить Source Hit@3", "constraints": ["Не менять embedding-модель", "Использовать heuristic reranker", "Сохранить разумное время ответа", "Сравнивать Baseline и Enhanced RAG", "Final Top-K 3"], "terms": {"качество": "Source Hit@3 retrieval"}, "confirmed_facts": [], "decisions": ["Сравнить threshold 0.30, 0.35 и 0.40"], "open_questions": [], "user_preferences": [], "version": 4, "updated_at": "2026-10-04T22:32:38.911078+00:00"}`

#### Ход 9

- Сообщение: Побочный вопрос: CrossEncoder обязателен?
- Standalone query: Побочный вопрос: CrossEncoder обязателен?
- Ожидаемые источники: ['README.md']
- Использованные источники: ['README.md']
- Состояние памяти: `{"goal": "Повысить Source Hit@3", "constraints": ["Не менять embedding-модель", "Использовать heuristic reranker", "Сохранить разумное время ответа", "Сравнивать Baseline и Enhanced RAG", "Final Top-K 3"], "terms": {"качество": "Source Hit@3 retrieval"}, "confirmed_facts": [], "decisions": ["Сравнить threshold 0.30, 0.35 и 0.40"], "open_questions": [], "user_preferences": [], "version": 4, "updated_at": "2026-10-04T22:32:38.911078+00:00"}`

#### Ход 10

- Сообщение: Исправление: разумное время ответа означает не более 5 секунд на retrieval, а не на всю генерацию.
- Standalone query: Исправление: разумное время ответа означает не более 5 секунд на retrieval, а не на всю генерацию.
- Ожидаемые источники: нет
- Использованные источники: нет
- Состояние памяти: `{"goal": "Повысить Source Hit@3", "constraints": ["Не менять embedding-модель", "Использовать heuristic reranker", "Сравнивать Baseline и Enhanced RAG", "Final Top-K 3", "Retrieval не более 5 секунд"], "terms": {"качество": "Source Hit@3 retrieval"}, "confirmed_facts": [], "decisions": ["Сравнить threshold 0.30, 0.35 и 0.40"], "open_questions": [], "user_preferences": [], "version": 5, "updated_at": "2026-10-04T22:32:38.944411+00:00"}`

#### Ход 11

- Сообщение: Вернёмся к основной метрике: как сравнить Source Hit@3 до и после reranking?
- Standalone query: Вернёмся к основной метрике: как сравнить Source Hit@3 до и после reranking?
- Ожидаемые источники: ['day23_reranking_filtering_omp.md']
- Использованные источники: ['day23_reranking_filtering_omp.md']
- Состояние памяти: `{"goal": "Повысить Source Hit@3", "constraints": ["Не менять embedding-модель", "Использовать heuristic reranker", "Сравнивать Baseline и Enhanced RAG", "Final Top-K 3", "Retrieval не более 5 секунд"], "terms": {"качество": "Source Hit@3 retrieval"}, "confirmed_facts": [], "decisions": ["Сравнить threshold 0.30, 0.35 и 0.40"], "open_questions": [], "user_preferences": [], "version": 5, "updated_at": "2026-10-04T22:32:38.944411+00:00"}`

#### Ход 12

- Сообщение: Дай итоговый план эксперимента с источниками и проверь, что Final Top-K 5 больше не считается активным.
- Standalone query: Дай итоговый план эксперимента с источниками и проверь, что Final Top-K 5 больше не считается активным.
- Ожидаемые источники: ['day23_reranking_filtering_omp.md']
- Использованные источники: ['day23_reranking_filtering_omp.md']
- Состояние памяти: `{"goal": "Повысить Source Hit@3", "constraints": ["Не менять embedding-модель", "Использовать heuristic reranker", "Сравнивать Baseline и Enhanced RAG", "Final Top-K 3", "Retrieval не более 5 секунд"], "terms": {"качество": "Source Hit@3 retrieval"}, "confirmed_facts": [], "decisions": ["Сравнить threshold 0.30, 0.35 и 0.40"], "open_questions": [], "user_preferences": [], "version": 5, "updated_at": "2026-10-04T22:32:38.944411+00:00"}`

## Итоговые метрики

- `turn_count`: 24
- `goal_preserved_share`: 1.000
- `constraints_preserved_share`: 1.000
- `cancelled_constraints_absent_share`: 1.000
- `follow_up_correct_share`: 0.958
- `source_hit_share`: 0.917
- `sources_block_share`: 1.000
- `task_state_correct_share`: 1.000
- `fabricated_sources_count`: 0
- `cancelled_constraint_returns`: 0
- `average_answer_elements_coverage`: 0.750
- `average_elapsed_seconds`: 0.004

## Вывод

Успех оценивается по сохранению цели и ограничений, корректности follow-up, источникам и отсутствию возвращённых отменённых значений — не только по факту генерации ответа.
