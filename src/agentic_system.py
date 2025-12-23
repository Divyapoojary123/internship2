import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Interfaces & Data Structures ---

@dataclass
class Message:
    role: str  # 'system', 'user', 'assistant'
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class Tool(ABC):
    """Abstract base class for tools that the agent can use."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def run(self, **kwargs) -> str:
        """Execute the tool logic."""
        pass

    def to_schema(self) -> Dict[str, Any]:
        """Return a JSON schema representation identifying the tool."""
        # Simplified schema for demonstration
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": {}} 
        }

class LLMProvider(ABC):
    """Abstract interface for a Large Language Model provider."""
    
    @abstractmethod
    def generate(self, messages: List[Message], tools: List[Tool]) -> str:
        pass

# --- Concrete Implementations ---

class MockLLM(LLMProvider):
    """A Mock LLM that simulates responses for demonstration purposes."""
    
    def generate(self, messages: List[Message], tools: List[Tool]) -> str:
        last_msg = messages[-1].content.lower()
        
        # Simple heuristic to simulate "thinking" and "tool calling"
        if "calculate" in last_msg and "result" not in last_msg:
            # Simulate a decision to use a tool
            return json.dumps({
                "action": "call_tool",
                "tool_name": "calculator",
                "parameters": {"expression": "2 + 2"}
            })
        elif "result" in last_msg:
            return "The answer is 4."
        else:
            return "I am ready to help. Please give me a task."

class CalculatorTool(Tool):
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Useful for performing mathematical calculations. Input should be a mathematical expression."

    def run(self, expression: str) -> str:
        try:
            # WARNING: eval is unsafe in production code; utilize safe math parsers instead.
            return str(eval(expression))
        except Exception as e:
            return f"Error calculating: {e}"

# --- The Agentic Core ---

class Agent:
    def __init__(self, name: str, llm: LLMProvider, tools: List[Tool]):
        self.name = name
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.memory: List[Message] = []
        
        # System prompt to define behavior
        self.memory.append(Message(
            role="system", 
            content=f"You are {name}, a helpful AI agent. You have access to the following tools: {', '.join(self.tools.keys())}."
        ))

    def chat(self, user_input: str) -> str:
        """Run a ReAct loop: Think -> Act -> Observe -> Response"""
        
        self.memory.append(Message(role="user", content=user_input))
        logger.info(f"User: {user_input}")

        max_steps = 5
        current_step = 0

        while current_step < max_steps:
            # 1. Ask LLM what to do
            response_text = self.llm.generate(self.memory, list(self.tools.values()))
            
            # 2. Check if it's a tool call (JSON format for this demo)
            try:
                # Heuristic: try to parse potential JSON action
                if response_text.strip().startswith("{") and "action" in response_text:
                    decision = json.loads(response_text)
                    
                    if decision.get("action") == "call_tool":
                        tool_name = decision.get("tool_name")
                        params = decision.get("parameters", {})
                        
                        logger.info(f"Agent decided to call tool: {tool_name} with {params}")
                        
                        # 3. Execute Tool
                        tool = self.tools.get(tool_name)
                        if tool:
                            result = tool.run(**params)
                            observation = f"Tool '{tool_name}' result: {result}"
                        else:
                            observation = f"Error: Tool '{tool_name}' not found."
                            
                        # 4. Update Memory with Observation
                        self.memory.append(Message(role="assistant", content=response_text))
                        self.memory.append(Message(role="system", content=observation)) # Treating observation as system info
                        
                        current_step += 1
                        continue # Go back to LLM with the new observation
                        
            except json.JSONDecodeError:
                pass # Not a JSON action, treat as final text
                
            # Final response case
            self.memory.append(Message(role="assistant", content=response_text))
            logger.info(f"Agent: {response_text}")
            return response_text

        return "I reached my step limit without a final answer."

# --- Usage Example ---

def main():
    # Initialize implementation
    llm = MockLLM()
    calculator = CalculatorTool()
    
    agent = Agent(
        name="AutoBot",
        llm=llm,
        tools=[calculator]
    )

    print("--- Starting Agentic Conversation ---")
    response = agent.chat("Can you calculate what is 2 + 2?")
    print(f"\nFinal Response: {response}")

if __name__ == "__main__":
    main()
