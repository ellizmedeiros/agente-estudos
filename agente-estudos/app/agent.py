# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import os
from pathlib import Path

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.environment import LocalEnvironment
from google.adk.models import Gemini
from google.adk.skills import load_skills_from_dir
from google.adk.tools.environment import EnvironmentToolset
from google.adk.tools.skill_toolset import SkillToolset
from google.genai import types

# Modelo obtido a partir da variável de ambiente MODEL
MODEL = os.getenv("MODEL", "gemini-3.8-flash")

# Caminhos do projeto
APP_DIR = Path(__file__).resolve().parent
SKILLS_DIR = APP_DIR / "skills"
PLANOS_DIR = APP_DIR.parent / "planos"
PLANOS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Carrega as skills usando SkillToolset
skills = load_skills_from_dir(SKILLS_DIR)
skill_toolset = SkillToolset(skills=skills)


# 2. Configura EnvironmentToolset com LocalEnvironment expondo apenas ReadFile e WriteFile
class RestrictedEnvironmentToolset(EnvironmentToolset):
    """EnvironmentToolset que expõe apenas as ferramentas filtradas (ReadFile e WriteFile)."""

    async def get_tools(self, readonly_context=None):
        tools = await super().get_tools(readonly_context)
        return [t for t in tools if self._is_tool_selected(t, readonly_context)]


environment_toolset = RestrictedEnvironmentToolset(
    environment=LocalEnvironment(working_dir=str(PLANOS_DIR)),
    tool_filter=["ReadFile", "WriteFile"],
)

# 3. Instrução do agente incluindo a data atual
hoje = datetime.date.today().strftime("%d/%m/%Y")

instruction = f"""Você é um mentor especialista em estudos. Sua missão é criar planos de estudo personalizados, com cronograma estruturado, metodologia de repetição espaçada e exportação para o Google Agenda.

Data de hoje: {hoje} (formato DD/MM/AAAA).

Siga rigorosamente as diretrizes e regras da skill plano-de-estudos:
1. Gating de informações: Antes de montar o plano, certifique-se de que a conversa possui todas as 4 informações obrigatórias (Objetivo, Prazo, Disponibilidade e Nível atual). Se faltar qualquer uma delas, faça uma única pergunta curta e amigável solicitando apenas o que falta. NÃO carregue recursos da skill nem monte o plano antes de obter essas informações.
2. Planejamento: Quando todas as informações estiverem presentes, use a ferramenta load_skill_resource para ler references/workflow.md e siga rigorosamente as etapas do workflow.
3. Salvamento da agenda: Ao finalizar o cronograma, utilize a ferramenta WriteFile para gravar o arquivo plano-de-estudos.csv na pasta do ambiente, respeitando as instruções de references/salvar_agenda.md e o formato do Google Agenda.
"""

root_agent = Agent(
    name="agente_estudos",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=instruction,
    tools=[skill_toolset, environment_toolset],
)

app = App(
    root_agent=root_agent,
    name="app",
)
