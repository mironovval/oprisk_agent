import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_gigachat import GigaChat
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import Tool
from langchain_core.messages import SystemMessage

# Загрузка переменных окружения
load_dotenv()

# Загрузка данных
def load_companies() -> List[Dict[str, Any]]:
    with open('data/companies.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def load_reviews() -> List[Dict[str, Any]]:
    with open('data/reviews_small.json', 'r', encoding='utf-8') as f:
        return json.load(f)

companies = load_companies()
reviews = load_reviews()

# Модель для ответа со списком компаний
class CompanyListResponse(BaseModel):
    companies: List[Dict[str, Any]] = Field(description="Список компаний с их основными данными")

# Модель для анализа компании
class CompanyAnalysisResponse(BaseModel):
    name: str = Field(description="Название компании")
    address: str = Field(description="Адрес компании")
    average_rating: float = Field(description="Средний рейтинг")
    risks: List[str] = Field(description="Список основных рисков")

# Инструмент для получения списка компаний
def get_companies(anyParm: str) -> str:
    """Возвращает список всех отделений/организаций в формате: ID Название Адрес"""
    # Формируем список в нужном формате
    companies_list = [
        f"{company['id']} {company['name']} {company['address']}"
        for company in companies
    ]
    # Возвращаем как строку с переносами
    return "\n".join(companies_list)

# Инструмент для анализа компании`
def analyze_company(company_id: str) -> str:
    """Анализирует компанию по её ID и возвращает результаты анализа"""
    orgId: int = int(company_id)
    company_reviews = [r for r in reviews if r['orgId'] == orgId]
    company = next((c for c in companies if c['id'] == orgId), {})

    if not company_reviews:
        return CompanyAnalysisResponse(
            name=company.get('name', 'Unknown'),
            address=company.get('address', 'Unknown'),
            average_rating=0,
            risks=["Нет данных об отзывах"]
        ).json()

    avg_rating = sum(r['rate'] for r in company_reviews) / len(company_reviews)
    negative_reviews = [r for r in company_reviews if r['rate'] <= 2]

    risks = list(set(
        r['comment'].split('.')[0].strip()
        for r in negative_reviews
    ))[:3] if negative_reviews else ["Не обнаружено значительных рисков"]

    return CompanyAnalysisResponse(
        name=company.get('name', 'Unknown'),
        address=company.get('address', 'Unknown'),
        average_rating=round(avg_rating, 2),
        risks=risks
    ).json()

# Создаем инструменты
@tool
def set_working_directory(directory: str, workflow: Optional[str] = None) -> Dict[str, str]:
    """Устанавливает рабочую директорию и фильтр по потоку"""
    print(f"\n>>> called set_working_directory(directory='{directory}', workflow='{workflow}')")
    global WORKING_DIRECTORY, CURRENT_WORKFLOW
    
    try:
        if not os.path.isdir(directory):
            return {"status": "error", "message": "Директория не существует"}
        
        WORKING_DIRECTORY = os.path.abspath(directory)
        CURRENT_WORKFLOW = workflow.lower() if workflow else None
        return {
            "status": "success", 
            "message": f"Директория: {WORKING_DIRECTORY}" + 
                      (f" | Фильтр: {workflow}" if workflow else "")
        }
    except Exception as e:
        return {"status": "error", "message": f"Ошибка: {str(e)}"}

def should_include_path(path: str) -> bool:
    """Проверяет, соответствует ли путь фильтру потока, всегда включая dag_utils.py"""
    path_str = str(path).lower()
    
    # Всегда включать dag_utils.py независимо от фильтра
    if "dag_utils.py" in path_str:
        return True
        
    if not CURRENT_WORKFLOW:
        return True
        
    return CURRENT_WORKFLOW in path_str

@tool
def list_python_files(subdirectory: Optional[str] = None) -> List[str]:
    """Список Python и SQL файлов с учётом фильтра потока (включая вложенные папки), файл functions/dag_utils.py относится ко всем потокам, всегда его учитывай"""
    print(f"\n>>> called list_python_files(subdirectory='{subdirectory}')")
    
    try:
        base_dir = Path(WORKING_DIRECTORY)
        target_dir = base_dir / subdirectory if subdirectory else base_dir
        
        if not target_dir.is_dir():
            return [f"Ошибка: Директория {target_dir} не существует"]
        
        files = []
        # Ищем как .py, так и .sql файлы
        for pattern in ["*.py", "*.sql"]:
            for path in target_dir.rglob(pattern):
                relative_path = str(path.relative_to(base_dir))
                
                # Фильтрация по пути
                if should_include_path(relative_path):
                    files.append(relative_path)
        files.append(relative_path)
        return files if files else ["Нет файлов, соответствующих фильтру"]
    
    except Exception as e:
        return [f"Ошибка: {str(e)}"]

@tool
def read_python_file(filename: str) -> str:
    """Читает Python или SQL файл, если он соответствует фильтру потока """
    print(f"\n>>> called read_python_file(filename='{filename}')")
    
    if CURRENT_WORKFLOW and not should_include_path(filename):
        return f"Файл не соответствует фильтру '{CURRENT_WORKFLOW}'"
    
    filepath = Path(WORKING_DIRECTORY) / filename
    try:
        content = filepath.read_text(encoding="utf-8")
        return content if content else "Файл пустой"
    except Exception as e:
        return f"Ошибка чтения: {str(e)}"

@tool
def summarize_python_file(filename: str) -> str:
    """Анализирует Python или SQL файл, если он соответствует фильтру потока"""
    print(f"\n>>> called summarize_python_file(filename='{filename}')")
    
    content = read_python_file.invoke(filename)
    if content.startswith("Ошибка") or content == "Файл пустой":
        return content
    
    try:
        file_type = "SQL" if filename.lower().endswith('.sql') else "Python"
        return giga.invoke(
            f"Детально проанализируй {file_type} код файла {filename}:\n\n{content}"
        )
    except Exception as e:
        return f"Ошибка анализа: {str(e)}"

@tool
def create_python_file(
    filename: str,
    code_content: str,
    subdirectory: Optional[str] = None,
    overwrite: bool = False,
    file_type: str = "python"  # Добавлен параметр для указания типа файла
) -> Dict[str, str]:
    """Создаёт Python или SQL файл с учётом структуры проекта"""
    print(f"\n>>> called create_python_file(filename='{filename}', file_type='{file_type}')")
    try:
        base_path = Path(WORKING_DIRECTORY)
        full_path = base_path / subdirectory / filename if subdirectory else base_path / filename
        
        # Определяем расширение файла в зависимости от типа
        if file_type.lower() == "sql" and not filename.lower().endswith('.sql'):
            full_path = full_path.with_suffix('.sql')
        elif file_type.lower() == "python" and not (filename.lower().endswith('.py') or filename.lower().endswith('.sql')):
            full_path = full_path.with_suffix('.py')
        
        if full_path.exists() and not overwrite:
            return {
                "status": "error",
                "message": f"Файл {filename} уже существует",
                "path": str(full_path.relative_to(base_path))
            }
        
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(code_content, encoding="utf-8")
        
        return {
            "status": "success",
            "message": f"Файл создан: {full_path.relative_to(base_path)}",
            "path": str(full_path.relative_to(base_path))
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Ошибка: {str(e)}",
            "path": str(full_path.relative_to(base_path)) if 'full_path' in locals() else filename
        }

tools = [
    set_working_directory,
    list_python_files,
    read_python_file,
    summarize_python_file,
    create_python_file
]

# Инициализация GigaChat
giga = GigaChat(
    credentials=os.getenv("GIGACHAT_CREDENTIALS"),
    model="GigaChat-Max",
    verify_ssl_certs=False
)

# Настройка промпта
prompt = f"""Ты файловый менеджер для Python и SQL проектов. Текущая рабочая директория: {WORKING_DIRECTORY}
Ты можешь:
1. Устанавливать рабочую директорию и поток (set_working_directory)
2. Перечислять Python и SQL файлы (list_python_files)
3. Читать содержимое файлов (read_python_file)
4. Анализировать код и просматривать его содержимое (summarize_python_file)
5. Создавать новые файлы (create_python_file) - укажи 'file_type="sql"' для SQL файлов

Все операции учитывают текущий фильтр по потоку: {CURRENT_WORKFLOW or 'нет фильтра'} Также всегда при проведении любой операции по потоку, учитывай файл
functions/dag_utils.py, который не содержит потока в названии, но относится ко всем потокам"""

# Создание агента
agent = create_openai_tools_agent(
    llm=giga,
    tools=tools,
    prompt=prompt
)

# Настройка исполнителя агента
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# Основной цикл
def main():
    print("Приветствую! Я агент-тестировщик.")

    while True:
        user_input = input("\nВаш запрос: ").strip()

        if user_input.lower() in ['выход', 'exit', 'quit']:
            print("До свидания!")
            break

        try:
            response = agent_executor.invoke({"input": user_input})
            print("\nОтвет:", response['output'])
        except Exception as e:
            print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    main()
