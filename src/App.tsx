import React, { useState, useEffect, useRef, useCallback } from "react";
import { DataSet, Network } from "vis-network/standalone";
import type { Node, Edge, Options, IdType } from "vis-network/standalone";
import "vis-network/styles/vis-network.css";
// DaisyUI is assumed to be included via CDN in the main HTML file

// Import Heroicons for simulation controls
import {
  PlayIcon,
  PauseIcon,
  BackwardIcon,
  ForwardIcon,
  ArrowPathIcon, // Icon for Reset Layout
  QuestionMarkCircleIcon, // Icon for the Guide
} from "@heroicons/react/24/solid";

// Interface for NFA states
interface State {
  id: number;
  isAccept: boolean;
}

// Interface for NFA transitions
interface Transition {
  from: State;
  to: State;
  symbol: string | undefined; // Changed from string | null to align with vis-network Edge label type
}

// Define the type for node updates, including the shadow property
// Updated NodeUpdateData['color'] to better match vis-network's Color type structure
type NodeUpdateData = {
  id: IdType;
  color?:
    | string
    | {
        border?: string;
        background?: string;
        highlight?: string | { border?: string; background?: string }; // highlight can be string or object
        hover?: string | { border?: string; background?: string }; // hover can be string or object
      };
  borderWidth?: number;
  shadow?:
    | boolean
    | {
        enabled?: boolean;
        color?: string;
        size?: number;
        x?: number;
        y?: number;
      };
};

// Helper function to check if a character is an operand (alphanumeric or escaped)
const isOperand = (c: string) => /^[a-zA-Z0-9]$/.test(c) || c === "ε"; // ε for explicit epsilon

// Helper function to check if a character is a unary operator
const isUnaryOperator = (c: string) => ["*", "+", "?"].includes(c);

// Preprocesses the regex string to insert explicit concatenation operators (·)
// Handles escaped characters like \*, \+, \?, \|, \(, \), \\, \e
const insertConcatenation = (regex: string): string => {
  let processedRegex = "";
  let prevChar = "";

  for (let i = 0; i < regex.length; i++) {
    const currentChar = regex[i];

    // Handle escape sequences first
    if (currentChar === "\\") {
      processedRegex += currentChar; // Add the backslash
      i++; // Move to the next character
      if (i < regex.length) {
        const escapedChar = regex[i];
        processedRegex += escapedChar; // Add the escaped character
        prevChar = escapedChar; // Update prevChar to the escaped character
      } else {
        throw new Error("Unterminated escape sequence at end of regex.");
      }
      continue; // Skip the rest of the loop for this iteration
    }

    // Insert concatenation based on the previous and current character
    // Check if prevChar exists and is not a control character that prevents concatenation
    if (
      prevChar &&
      prevChar !== "|" && // Not after alternation
      currentChar !== "|" && // Not before alternation
      // --- MODIFIED: Exclude insertion before unary operators (*, +, ?) ---
      currentChar !== "*" &&
      currentChar !== "+" &&
      currentChar !== "?" &&
      // --- END MODIFIED ---
      currentChar !== ")" && // Not before closing parenthesis
      prevChar !== "(" // Not after opening parenthesis
    ) {
      // Conditions for implicit concatenation:
      // 1. Operand followed by Operand (ab)
      // 2. Operand followed by Opening Parenthesis (a(b|c))
      // 3. Closing Parenthesis followed by Operand ((a|b)c)
      // 4. Closing Parenthesis followed by Opening Parenthesis ((a)(b))
      // 5. Unary Operator (*, +, ?) followed by Operand (a*b)
      // 6. Unary Operator (*, +, ?) followed by Opening Parenthesis (a*(b|c))
      // 7. Escaped character followed by Operand (\ab)
      // 8. Escaped character followed by Opening Parenthesis (\a(b|c))

      const needsConcat =
        (isOperand(prevChar) || // Operand
          prevChar === ")" || // Closing parenthesis
          isUnaryOperator(prevChar) || // Unary operator
          (prevChar.length === 1 &&
            prevChar !== "\\" &&
            regex[i - 2] === "\\")) && // Previous char was escaped (and not the backslash itself)
        (isOperand(currentChar) || // Operand
          currentChar === "(" || // Opening parenthesis
          currentChar === "\\"); // Start of an escape sequence

      if (needsConcat) {
        processedRegex += "·";
      }
    }

    processedRegex += currentChar;
    prevChar = currentChar;
  }
  return processedRegex;
};

// Parses a regular expression string to its postfix notation using the Shunting-Yard algorithm.
// Inserts explicit concatenation operators first.
// Throws an error for invalid regex syntax.
const parseRegexToPostfix = (regex: string): string => {
  if (regex.length === 0) throw new Error("Empty regular expression");

  // Preprocess to insert explicit concatenation
  let processedRegex;
  try {
    processedRegex = insertConcatenation(regex);
  } catch (error) {
    if (error instanceof Error) {
      throw new Error(`Preprocessing error: ${error.message}`);
    }
    throw new Error("Preprocessing failed.");
  }

  const precedence: { [key: string]: number } = {
    "|": 1, // Alternation
    "·": 2, // Concatenation
    "?": 3, // Zero or one
    "*": 3, // Zero or more (Kleene star)
    "+": 3, // One or more (Kleene plus)
  };

  const outputQueue: string[] = [];
  const operatorStack: string[] = [];
  let parenCount = 0;

  // Validate processed regex start and end after concatenation insertion
  const firstChar = processedRegex[0];
  if (["|", "*", "+", "?", "·"].includes(firstChar)) {
    throw new Error(
      `Invalid start of expression after preprocessing: '${firstChar}'.`
    );
  }
  const lastChar = processedRegex[processedRegex.length - 1];
  if (["|", "·", "("].includes(lastChar)) {
    throw new Error(
      `Invalid end of expression after preprocessing: '${lastChar}'.`
    );
  }

  let i = 0;
  while (i < processedRegex.length) {
    const char = processedRegex[i];

    if (char === "\\") {
      // Handle escaped characters during parsing
      i++; // Move to the next character after '\'
      if (i < processedRegex.length) {
        const escapedChar = processedRegex[i];
        if (escapedChar === "e") {
          outputQueue.push("ε"); // Push the epsilon symbol
        } else {
          outputQueue.push(escapedChar); // Push other escaped characters
        }
        // i is already incremented once for '\', the loop's i++ handles the second increment
      } else {
        throw new Error("Unterminated escape sequence at end of regex.");
      }
      i++; // Increment i again to skip the escaped character
      continue;
    }

    if (isOperand(char)) {
      outputQueue.push(char);
    } else {
      switch (char) {
        case "(":
          parenCount++;
          operatorStack.push("(");
          break;
        case ")":
          if (parenCount === 0)
            throw new Error(`Unmatched closing parenthesis at position ${i}.`);
          parenCount--;
          while (
            operatorStack.length > 0 &&
            operatorStack[operatorStack.length - 1] !== "("
          ) {
            outputQueue.push(operatorStack.pop()!);
          }
          if (operatorStack.length === 0)
            throw new Error(
              `Mismatched parentheses; stack empty when expecting '('. Position ${i}`
            );
          operatorStack.pop(); // Pop '('
          break;
        case "|":
        case "·":
        case "*":
        case "+":
        case "?":
          // Handle operator precedence
          while (
            operatorStack.length > 0 &&
            operatorStack[operatorStack.length - 1] !== "(" &&
            precedence[char] <=
              precedence[operatorStack[operatorStack.length - 1]]
          ) {
            outputQueue.push(operatorStack.pop()!);
          }
          operatorStack.push(char);
          break;
        default:
          // Should not happen if insertConcatenation and isOperand are correct,
          // but as a fallback for any unhandled characters.
          throw new Error(
            `Invalid or unhandled character '${char}' at position ${i}.`
          );
      }
    }
    i++;
  }

  if (parenCount > 0)
    throw new Error("Unmatched opening parenthesis in the expression.");

  // Pop any remaining operators from the stack to the output queue
  while (operatorStack.length > 0) {
    const op = operatorStack.pop()!;
    if (op === "(")
      throw new Error(
        "Mismatched parentheses (remaining '(' on stack at end of parsing)."
      );
    outputQueue.push(op);
  }

  return outputQueue.join("");
};

// Builds an NFA from a postfix regular expression using Thompson's construction algorithm.
const buildNFAFromPostfix = (postfix: string) => {
  let stateCounter = 0;
  // Corrected: Explicitly type the stack
  const stack: Array<{ start: State; end: State; transitions: Transition[] }> =
    [];

  const createState = (isAccept = false): State => ({
    id: stateCounter++,
    isAccept,
  });

  const addTransition = (
    from: State,
    to: State,
    symbol: string | undefined
  ): Transition => ({
    from,
    to,
    symbol,
  });

  // Handle empty regex case - represents the empty string
  if (postfix.length === 0) {
    const start = createState(true); // Start is also the accept state for empty string
    return { start, end: start, transitions: [] }; // NFA for empty string
  }

  for (const char of postfix) {
    switch (char) {
      case "·": {
        // Concatenation: NFA1 -> NFA2
        if (stack.length < 2)
          throw new Error(
            "Invalid postfix: Not enough operands for '·' (concatenation)."
          );
        const frag2 = stack.pop()!;
        const frag1 = stack.pop()!;
        // Connect end of frag1 to start of frag2 with epsilon
        // The start of the new NFA is frag1's start, the end is frag2's end.
        // frag1's end is no longer an accept state in the combined NFA.
        frag1.end.isAccept = false;
        // frag2's start is not an accept state in the combined NFA (it's an intermediate state).
        frag2.start.isAccept = false;
        const transitions = [
          ...frag1.transitions,
          ...frag2.transitions,
          addTransition(frag1.end, frag2.start, undefined), // Epsilon transition (undefined symbol)
        ];
        stack.push({ start: frag1.start, end: frag2.end, transitions });
        break;
      }
      case "|": {
        // Alternation: New Start -> NFA1 | NFA2 -> New End
        if (stack.length < 2)
          throw new Error(
            "Invalid postfix: Not enough operands for '|' (alternation)."
          );
        const fragB = stack.pop()!;
        const fragA = stack.pop()!;
        const start = createState(); // New start state
        const end = createState(true); // New accept state

        // Original fragment end states are no longer accept states
        fragA.end.isAccept = false;
        fragB.end.isAccept = false;
        // Original fragment start states are not accept states (they are connected from new start)
        fragA.start.isAccept = false;
        fragB.start.isAccept = false;

        const transitions = [
          ...fragA.transitions,
          ...fragB.transitions,
          addTransition(start, fragA.start, undefined), // Epsilon from new start to fragA start
          addTransition(start, fragB.start, undefined), // Epsilon from new start to fragB start
          addTransition(fragA.end, end, undefined), // Epsilon from fragA end to new end
          addTransition(fragB.end, end, undefined), // Epsilon from fragB end to new end
        ];
        stack.push({ start, end, transitions });
        break;
      }
      case "*": {
        // Kleene Star: New Start -> NFA -> New End, with loops
        if (stack.length < 1)
          throw new Error(
            "Invalid postfix: Not enough operands for '*' (Kleene star)."
          );
        const frag = stack.pop()!;
        const start = createState(); // New start state
        const end = createState(true); // New accept state (for zero occurrences)

        // Original fragment end state is no longer accept
        frag.end.isAccept = false;
        // Original fragment start state is not accept
        frag.start.isAccept = false;

        const transitions = [
          ...frag.transitions,
          addTransition(start, frag.start, undefined), // Epsilon from new start to frag start
          addTransition(frag.end, end, undefined), // Epsilon from frag end to new end
          addTransition(frag.end, frag.start, undefined), // Epsilon loop from frag end to frag start (for one or more)
          addTransition(start, end, undefined), // Epsilon from new start to new end (for zero occurrences)
        ];
        stack.push({ start, end, transitions });
        break;
      }
      case "+": {
        // Kleene Plus: New Start -> NFA -> New End, with loop (requires at least one match)
        if (stack.length < 1)
          throw new Error(
            "Invalid postfix: Not enough operands for '+' (Kleene plus)."
          );
        const frag = stack.pop()!;
        const start = createState(); // New start state
        const end = createState(true); // New accept state

        // Original fragment end state is no longer accept
        frag.end.isAccept = false;
        // Original fragment start state is not accept
        frag.start.isAccept = false;

        const transitions = [
          ...frag.transitions,
          addTransition(start, frag.start, undefined), // Epsilon from new start to frag start
          addTransition(frag.end, end, undefined), // Epsilon from frag end to new end
          addTransition(frag.end, frag.start, undefined), // Epsilon loop from frag end to frag start (for one or more)
          // No direct epsilon from start to end, as it requires at least one match
        ];
        stack.push({ start, end, transitions });
        break;
      }
      case "?": {
        // Zero or One: New Start -> NFA -> New End, with optional epsilon path
        if (stack.length < 1)
          throw new Error(
            "Invalid postfix: Not enough operands for '?' (zero or one)."
          );
        const frag = stack.pop()!;
        const start = createState(); // New start state
        const end = createState(true); // New accept state (for zero occurrences)

        // Original fragment end state is no longer accept
        frag.end.isAccept = false;
        // Original fragment start state is not accept
        frag.start.isAccept = false;

        const transitions = [
          ...frag.transitions,
          addTransition(start, frag.start, undefined), // Epsilon from new start to frag start
          addTransition(frag.end, end, undefined), // Epsilon from frag end to new end
          addTransition(start, end, undefined), // Epsilon from new start to new end (for zero occurrences)
        ];
        stack.push({ start, end, transitions });
        break;
      }
      case "ε": // Explicit epsilon operand
      default: {
        // Literal character or Epsilon ('ε')
        const start = createState();
        const end = createState(true);
        const symbol = char === "ε" ? undefined : char; // Use undefined for epsilon symbol
        stack.push({
          start,
          end,
          transitions: [addTransition(start, end, symbol)],
        });
        break;
      }
    }
  }

  if (stack.length !== 1)
    throw new Error(
      "Invalid postfix expression or construction error: Stack should contain exactly one NFA fragment at the end."
    );
  const finalNFA = stack.pop()!;
  // Ensure the final end state is marked as accept
  finalNFA.end.isAccept = true;
  return finalNFA;
};

// Computes the epsilon closure for a set of states.
const computeEpsilonClosure = (
  states: Set<State>,
  transitions: Transition[]
): Set<State> => {
  const closure = new Set<State>();
  const stack: State[] = [];

  // Initialize closure and stack with the starting states
  states.forEach((s) => {
    closure.add(s);
    stack.push(s);
  });

  // Explore reachable states via epsilon transitions
  while (stack.length > 0) {
    const current = stack.pop()!;
    transitions
      .filter((t) => t.from.id === current.id && t.symbol === undefined) // Find epsilon transitions from the current state (symbol is undefined)
      .forEach((t) => {
        // Use Set.has() for efficient membership check
        if (!closure.has(t.to)) {
          closure.add(t.to);
          stack.push(t.to); // Add new states to the stack for further exploration
        }
      });
  }
  return closure;
};

// Processes a single character step in the NFA simulation.
const processCharacterStep = (
  activeStates: Set<State>,
  char: string,
  allTransitions: Transition[]
): Set<State> => {
  const directNextStates = new Set<State>();
  // Find states reachable directly by the character transition
  activeStates.forEach((s) => {
    allTransitions
      .filter((t) => t.from.id === s.id && t.symbol === char)
      .forEach((t) => {
        directNextStates.add(t.to);
      });
  });
  // Compute the epsilon closure of the direct next states
  return computeEpsilonClosure(directNextStates, allTransitions);
};

const App: React.FC = () => {
  const [localRegexInput, setLocalRegexInput] = useState("");
  const [testString, setTestString] = useState("");
  const [nfa, setNFA] = useState<{
    start: State;
    end: State;
    transitions: Transition[];
  } | null>(null);
  const [nfaAllStates, setNfaAllStates] = useState<State[]>([]);
  const [history, setHistory] = useState<Set<State>[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isGuideOpen, setIsGuideOpen] = useState(true);
  const [animationSpeed, setAnimationSpeed] = useState(1000); // Milliseconds per step
  const [currentTheme, setCurrentTheme] = useState("light"); // State to track current theme

  const networkRef = useRef<Network | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const nodesRef = useRef<DataSet<Node> | null>(null);
  const edgesRef = useRef<DataSet<Edge> | null>(null);
  // Corrected type for originalNodeColors
  const originalNodeColors = useRef<Map<IdType, Node["color"] | undefined>>(
    new Map()
  ); // To store original colors for highlighting

  // Example regexes for quick selection
  const exampleRegexes = [
    "", // Empty string
    "a",
    "a|b",
    "a·b", // Explicit concatenation
    "a*",
    "a+",
    "a?",
    "a(b|c)*d",
    "(a|b)+c?",
    "\\*", // Escaped characters
    "\\+",
    "\\?",
    "\\|",
    "\\(",
    "\\)",
    "\\\\",
    "\\e", // Explicit epsilon
    "(a\\|b)*", // Regex with escaped metacharacters
  ];

  // Effect to set initial theme from localStorage or default
  useEffect(() => {
    const savedTheme = localStorage.getItem("theme") || "light";
    document.documentElement.setAttribute("data-theme", savedTheme);
    setCurrentTheme(savedTheme);
  }, []); // Run only once on mount

  // Effect to update nfaAllStates and reset history when NFA changes
  useEffect(() => {
    if (nfa) {
      const statesMap = new Map<number, State>();
      // Collect all states from transitions
      nfa.transitions.forEach((t) => {
        statesMap.set(t.from.id, t.from);
        statesMap.set(t.to.id, t.to);
      });
      // Ensure start and end states are included, even if they have no transitions
      statesMap.set(nfa.start.id, nfa.start);
      statesMap.set(nfa.end.id, nfa.end);

      const uniqueStatesFromNFA = Array.from(statesMap.values()).sort(
        (a, b) => a.id - b.id
      ); // Sort by ID for consistency
      setNfaAllStates(uniqueStatesFromNFA);

      // Compute initial epsilon closure for the start state
      const initialActiveStates = computeEpsilonClosure(
        new Set([nfa.start]),
        nfa.transitions
      );
      setHistory([initialActiveStates]);
      setCurrentStep(0);
    } else {
      // Reset states and history if NFA is null
      setNfaAllStates([]);
      setHistory([]);
      setCurrentStep(0);
    }
  }, [nfa]); // Dependency on nfa

  // Function to highlight states by changing background color and adding a subtle pulse/shadow
  const highlightStates = useCallback(
    (statesToHighlight: Set<State>) => {
      if (!nodesRef.current || nfaAllStates.length === 0) {
        // If NFA is not available or states are not loaded, ensure nodes are reset
        if (nodesRef.current && nodesRef.current.length > 0) {
          const resetUpdates: NodeUpdateData[] = nodesRef.current
            .getIds()
            .map((nodeId) => ({
              id: nodeId,
              color: originalNodeColors.current.get(nodeId) || "#6b7280", // Revert to original or default gray
              shadow: false, // Ensure shadow is off
            }));
          if (resetUpdates.length > 0) {
            nodesRef.current.update(resetUpdates);
          }
        }
        return;
      }

      // Create updates array for vis-network
      const updates: NodeUpdateData[] = nfaAllStates
        .map((s) => {
          const isActive = statesToHighlight.has(s); // Use Set.has() for efficient check
          const originalColor = originalNodeColors.current.get(s.id);

          // Construct the color object based on the updated NodeUpdateData type
          const updatedColor: NodeUpdateData["color"] = isActive
            ? { background: "#FDE047", border: "#D97706" } // Yellow background for active states
            : originalColor || { background: "#6b7280", border: "#4b5563" }; // Revert to original or default gray

          return {
            id: s.id,
            color: updatedColor,
            // Add a temporary shadow for active states for visual feedback
            shadow: isActive
              ? { enabled: true, color: "rgba(253, 224, 71, 0.8)", size: 10 } // Yellow shadow
              : false, // No shadow for inactive states
          };
        })
        .filter((update) => update !== null) as NodeUpdateData[]; // Filter out null if any logic error occurred

      if (updates.length > 0) {
        nodesRef.current.update(updates);

        // Optional: Remove the shadow after a short delay to create a pulse effect
        if (
          updates.some(
            (u) => u.shadow && typeof u.shadow === "object" && u.shadow.enabled
          )
        ) {
          setTimeout(() => {
            const shadowOffUpdates: NodeUpdateData[] = updates.map(
              (update) => ({
                id: update.id,
                shadow: false, // Turn shadow off
              })
            );
            nodesRef.current?.update(shadowOffUpdates);
          }, 300); // Shadow lasts for 300ms
        }
      }
    },
    [nfaAllStates] // Dependency on nfaAllStates to ensure updates apply after states are set
  );

  // Effect to draw/update network. Only depends on NFA structure and state styling logic.
  useEffect(() => {
    if (nfa && containerRef.current && nfaAllStates.length > 0) {
      // Destroy existing network if it exists
      if (networkRef.current) {
        networkRef.current.destroy();
        networkRef.current = null;
      }

      originalNodeColors.current.clear(); // Clear previous colors

      // Define node styling based on state type (Start, Accept, Normal)
      const visNodes = new DataSet<Node>(
        nfaAllStates.map((s) => {
          const isStart = s.id === nfa.start.id;
          const isAccept = s.isAccept; // Use the isAccept property directly from the State object

          const nodeColor = {
            background: isStart
              ? "#3b82f6" // blue-500
              : isAccept
              ? "#10b981" // emerald-500
              : "#6b7280", // gray-500
            border: isStart ? "#2563eb" : isAccept ? "#059669" : "#4b5563", // Darker shades for borders
            highlight: {
              background: isStart
                ? "#60a5fa" // blue-400
                : isAccept
                ? "#34d399" // emerald-400
                : "#9ca3af", // gray-400
              border: isStart ? "#2563eb" : isAccept ? "#059669" : "#4b5563",
            },
            hover: {
              background: isStart
                ? "#60a5fa" // blue-400
                : isAccept
                ? "#34d399" // emerald-400
                : "#9ca3af", // gray-400
              border: isStart ? "#2563eb" : isAccept ? "#059669" : "#4b5563",
            },
          };

          originalNodeColors.current.set(s.id, nodeColor); // Store original color

          return {
            id: s.id,
            label: isStart ? "Start" : isAccept ? "Accept" : `${s.id}`, // Use ID for normal states
            color: nodeColor,
            shape: "circle", // Consistent circle shape
            size: 30,
            font: { color: "#ffffff", size: 16, face: "Arial" }, // White text for labels
            borderWidth: isStart ? 3 : 2,
            shadow: false, // Initial state has no shadow
            title: `State ID: ${s.id}${s.isAccept ? " (Accept State)" : ""}${
              isStart ? " (Start State)" : ""
            }`, // Tooltip for states using vis-network's built-in tooltip via title
          };
        })
      );

      // Define edge styling
      const visEdgesData = new DataSet<Edge>(
        nfa.transitions.map((t, index) => {
          // Create DataSet directly here
          const isEpsilon = t.symbol === undefined; // Check for undefined symbol
          // Define smooth options based on vis-network's Edge type
          const smoothOptions: Edge["smooth"] = isEpsilon
            ? { enabled: true, type: "curvedCW", roundness: 0.2 } // Curved for epsilon
            : { enabled: true, type: "continuous", roundness: 0 }; // Straight for others

          // Simple color coding for edges (can be expanded)
          let edgeColor = "#4b5563"; // Default gray
          if (t.symbol === "a") edgeColor = "#ef4444"; // Red for 'a'
          if (t.symbol === "b") edgeColor = "#3b82f6"; // Blue for 'b'
          if (t.symbol === "c") edgeColor = "#22c593"; // Green for 'c' (using emerald-500)
          if (isEpsilon) edgeColor = "#a855f7"; // Purple for epsilon

          return {
            id: `e${index}`,
            from: t.from.id,
            to: t.to.id,
            label: isEpsilon ? "ε" : t.symbol, // Label is always a string
            arrows: "to",
            dashes: isEpsilon, // Dashed lines for epsilon transitions
            color: { color: edgeColor, highlight: "#1f2937", hover: "#1f2937" }, // Edge color
            font: {
              size: 14,
              align: "middle",
              color: "#111827", // Dark text for labels
              strokeWidth: 0,
            }, // Edge label styling
            smooth: smoothOptions,
            width: 2, // Default edge width
            hoverWidth: 3, // Thicker on hover
            selectionWidth: 3, // Thicker when selected
            title: `Transition: ${isEpsilon ? "ε" : t.symbol || "ε"}`, // Tooltip for edges using vis-network's built-in tooltip
          };
        })
      );

      // Assign the created DataSets directly to the refs
      edgesRef.current = visEdgesData;
      nodesRef.current = visNodes;

      // Define network options - Updated layout options
      const options: Options = {
        nodes: {
          shape: "circle", // Consistent circle shape
          size: 30,
          font: {
            size: 16,
            color: "#ffffff", // White text for labels
            face: "Arial",
          },
          borderWidth: 2,
          shadow: {
            enabled: false, // Shadow disabled by default
            color: "rgba(255, 0, 0, 0.8)", // Red shadow color (default, overridden by highlightStates)
            size: 15, // Shadow size
            x: 0,
            y: 0,
          },
        },
        edges: {
          arrows: {
            to: { enabled: true, scaleFactor: 1 }, // Ensure arrowheads are enabled
          },
          smooth: {
            type: "cubicBezier", // Smooth edges by default
            roundness: 0.4,
            enabled: true, // Ensure smooth is enabled
          },
          font: {
            size: 14,
            color: "#111827", // Edge label color
            strokeWidth: 0,
            align: "middle",
          },
          color: { color: "#4b5563", highlight: "#1f2937", hover: "#1f2937" },
          width: 2,
          hoverWidth: 3,
          selectionWidth: 3,
        },
        physics: {
          enabled: true, // Enable physics for a force-directed layout
          // You can fine-tune physics options here for better layout
          solver: "barnesHut", // or 'repulsion', 'hierarchicalRepulsion'
          barnesHut: {
            gravitationalConstant: -2000, // Negative for repulsion
            centralGravity: 0.3,
            springLength: 95,
            springConstant: 0.04,
            damping: 0.09,
            avoidOverlap: 0.5,
          },
          stabilization: { iterations: 1000 }, // Run stabilization to settle the layout initially
        },
        layout: {
          hierarchical: {
            enabled: false, // Disable hierarchical layout
            // If you want to try the hybrid approach, enable hierarchical here
            // and then call network.stabilize() after the network is created.
            // enabled: true,
            // direction: "LR",
            // sortMethod: "directed",
            // levelSeparation: 100,
            // nodeSpacing: 50,
            // treeSpacing: 100,
          },
        },
        interaction: {
          dragNodes: true, // Allow dragging nodes - This is already enabled!
          dragView: true, // Allow dragging the view
          zoomView: true, // Allow zooming
          hover: true, // Enable hover effects
          navigationButtons: true, // Add navigation buttons
          keyboard: true, // Enable keyboard navigation
          tooltipDelay: 100, // Delay for tooltips
        },
      };

      // Create the network instance
      networkRef.current = new Network(
        containerRef.current!,
        { nodes: nodesRef.current, edges: edgesRef.current },
        options
      );

      // Note: vis-network handles tooltips automatically if the 'title' property is set on nodes/edges.
      // No need for custom hover events or tooltip state for basic tooltips.

      // Initial highlight is handled by the dedicated highlight effect below

      // If using a hybrid layout (hierarchical + stabilize), call stabilize here:
      // networkRef.current.stabilize();
    } else if (!nfa && networkRef.current) {
      // Clean up network if NFA is removed
      networkRef.current.destroy();
      networkRef.current = null;
      nodesRef.current?.clear();
      edgesRef.current?.clear();
      originalNodeColors.current.clear();
    }

    // Cleanup function to destroy network on component unmount
    return () => {
      if (networkRef.current) {
        networkRef.current.destroy();
        networkRef.current = null;
      }
    };
  }, [nfa, nfaAllStates]); // Dependencies on nfa and nfaAllStates

  // Handle conversion of regex to NFA
  const handleConvert = () => {
    setError(null);
    // Clear previous network and data
    if (networkRef.current) {
      networkRef.current.destroy();
      networkRef.current = null;
    }
    // Clear DataSets explicitly
    nodesRef.current = null;
    edgesRef.current = null;
    originalNodeColors.current.clear();

    setNFA(null); // This will trigger the useEffect to clean up and reset state
    setTestString(""); // Clear test string on new NFA
    setHistory([]);
    setCurrentStep(0);
    setIsPlaying(false);

    try {
      if (!localRegexInput.trim()) {
        throw new Error("Please enter a regular expression");
      }
      // Parse regex and build NFA
      const postfix = parseRegexToPostfix(localRegexInput);
      const newNfa = buildNFAFromPostfix(postfix);
      setNFA(newNfa);
    } catch (err) {
      // Handle errors during parsing or construction
      setError(
        err instanceof Error
          ? err.message
          : "Invalid regular expression or NFA construction failed"
      );
      setNFA(null); // Ensure NFA is null on error
    }
  };

  // Handle stepping forward in the simulation
  const handleStepForward = useCallback(() => {
    // Check if NFA exists, not at the end of the test string, and history is available
    if (!nfa || currentStep >= testString.length || !history[currentStep])
      return;

    const currentActiveStates = history[currentStep];
    const nextCharacter = testString[currentStep];

    // Process the next character and compute the next set of active states
    const nextActiveStates = processCharacterStep(
      currentActiveStates,
      nextCharacter,
      nfa.transitions
    );

    // Update history with the new set of active states
    const newHistory = [...history.slice(0, currentStep + 1), nextActiveStates];
    setHistory(newHistory);
    setCurrentStep(currentStep + 1); // Move to the next step
  }, [nfa, currentStep, testString, history]); // Dependencies

  // Handle stepping backward in the simulation
  const handleStepBackward = useCallback(() => {
    // Allow stepping back as long as not at the beginning
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1); // Move to the previous step
      // History is kept, so no need to slice
    }
  }, [currentStep]); // Dependency on currentStep

  // Handle resetting the layout (Stabilize and Fit)
  const handleResetLayout = () => {
    if (networkRef.current) {
      // Stabilize for a short duration to arrange nodes
      networkRef.current.stabilize(100); // Stabilize for 100ms
      // Fit the network to the view after a slight delay to allow stabilization to start
      setTimeout(() => {
        networkRef.current?.fit();
      }, 150); // Delay fit slightly
    }
  };

  // Effect to highlight states when currentStep or history changes.
  useEffect(() => {
    // Highlight states if NFA exists and history is available for the current step
    if (nfa && history.length > 0 && history[currentStep] !== undefined) {
      highlightStates(history[currentStep]);
    } else if (!nfa && nodesRef.current && nodesRef.current.length > 0) {
      // If NFA is cleared, ensure all highlighting is removed
      const resetUpdates: NodeUpdateData[] = nodesRef.current
        .getIds()
        .map((nodeId) => ({
          id: nodeId,
          color: originalNodeColors.current.get(nodeId) || "#6b7280", // Revert to original or default gray
          shadow: false, // Ensure shadow is off
        }));
      if (resetUpdates.length > 0) {
        nodesRef.current.update(resetUpdates);
      }
    }
  }, [currentStep, history, nfa, highlightStates]); // Dependencies

  // Effect for automatic play functionality
  useEffect(() => {
    let timeoutId: number;
    // If playing, NFA exists, and not at the end of the string, step forward
    if (isPlaying && nfa && currentStep < testString.length) {
      timeoutId = window.setTimeout(handleStepForward, animationSpeed); // Step based on animationSpeed
    } else if (isPlaying && currentStep >= testString.length) {
      // Stop playing when the end of the string is reached
      setIsPlaying(false);
    }
    // Cleanup function to clear the timeout
    return () => clearTimeout(timeoutId);
  }, [
    isPlaying,
    nfa,
    currentStep,
    testString.length,
    handleStepForward,
    animationSpeed,
  ]); // Dependencies

  // Reset simulation when test string changes or NFA is built/changed
  useEffect(() => {
    if (nfa) {
      // Recompute initial epsilon closure for the start state
      const initialActiveStates = computeEpsilonClosure(
        new Set([nfa.start]),
        nfa.transitions
      );
      setHistory([initialActiveStates]); // Reset history
      setCurrentStep(0); // Reset step counter
      setIsPlaying(false); // Pause playback on string change or new NFA
    }
  }, [testString, nfa]); // Dependencies on testString and nfa

  // Add keyboard shortcuts for simulation controls (Space, Arrows)
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Prevent actions if input fields are focused
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement
      ) {
        return;
      }

      if (e.key === " ") {
        e.preventDefault(); // Prevent space from scrolling
        if (nfa && testString && currentStep < testString.length) {
          setIsPlaying((prev) => !prev);
        }
      } else if (e.key === "ArrowRight") {
        if (!isPlaying) handleStepForward();
      } else if (e.key === "ArrowLeft") {
        if (!isPlaying) handleStepBackward();
      }
    };
    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
  }, [
    isPlaying,
    nfa,
    testString,
    currentStep,
    handleStepForward,
    handleStepBackward,
  ]); // Dependencies

  // Add specific keydown handler for the regex input to prevent browser zoom on '+'
  const handleRegexInputKeyDown = (
    e: React.KeyboardEvent<HTMLInputElement>
  ) => {
    if (e.key === "+") {
      // Prevent default browser behavior (like zooming)
      e.preventDefault();
      // Manually insert the '+' character into the input value
      const input = e.target as HTMLInputElement;
      const { selectionStart, selectionEnd, value } = input;
      const newValue =
        value.substring(0, selectionStart!) +
        "+" +
        value.substring(selectionEnd!);
      setLocalRegexInput(newValue);

      // Manually set the cursor position after insertion
      // Use setTimeout to allow the state update to potentially render first
      setTimeout(() => {
        input.setSelectionRange(selectionStart! + 1, selectionStart! + 1);
      }, 0);
    }
  };

  // Check for acceptance only when the simulation is at the end of the string
  const isAccepted =
    currentStep === testString.length && history.length > currentStep // Ensure currentStep is valid index in history
      ? Array.from(history[currentStep]).some((s) => s.isAccept) // Check if any active state is an accept state
      : false;

  // Textual summary of the NFA
  const nfaSummary = nfa ? (
    <div className="text-sm text-base-content space-y-1">
      <p>
        <span className="font-semibold">NFA Summary:</span>
      </p>
      <p>Total States: {nfaAllStates.length}</p>
      <p>Start State ID: {nfa.start.id}</p>
      <p>
        Accept State IDs:{" "}
        {nfaAllStates
          .filter((s) => s.isAccept)
          .map((s) => s.id)
          .join(", ") || "None"}
      </p>
      <p>Total Transitions: {nfa.transitions.length}</p>
      {nfaAllStates.length > 50 && (
        <p className="text-warning font-semibold">
          Warning: This NFA is large and may impact performance.
        </p>
      )}
    </div>
  ) : null;

  // Function to handle theme change and save to localStorage
  const handleThemeChange = (theme: string) => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme); // Save theme preference
    setCurrentTheme(theme); // Update state
  };

  return (
    // Apply theme attribute to the root element (or body in index.html) for global styling
    // Removed data-theme="light" from this div to allow the document.documentElement setting to control it
    <div className="min-h-screen bg-base-200 flex flex-col">
      {/* Global Header */}
      <div className="navbar bg-base-100 shadow-md">
        <div className="container mx-auto px-4 sm:px-8">
          <div className="flex-1">
            <a className="btn btn-ghost text-xl normal-case text-base-content">
              Regex to NFA Visualizer
            </a>
          </div>
          <div className="flex-none gap-2">
            {/* DaisyUI Theme Switcher */}
            <div className="z-10">
              {" "}
              {/* Ensure theme switcher is above other content */}
              <select
                data-choose-theme // DaisyUI attribute for theme selection
                className="select select-sm select-bordered"
                onChange={(e) => handleThemeChange(e.target.value)}
                value={currentTheme} // Control the select value with state
                aria-label="Select Theme"
              >
                <option value="light">🌞 Light</option>
                <option value="dark">🌙 Dark</option>
                <option value="cupcake">🧁 Cupcake</option>
                <option value="bumblebee">🐝 Bumblebee</option>
                <option value="emerald">🌿 Emerald</option>
                <option value="corporate">🏢 Corporate</option>
                <option value="synthwave">🌃 Synthwave</option>
                <option value="retro">💾 Retro</option>
                <option value="cyberpunk">🤖 Cyberpunk</option>
                <option value="valentine">🌸 Valentine</option>
                <option value="halloween">🎃 Halloween</option>
                <option value="garden">🏡 Garden</option>
                <option value="forest">🌲 Forest</option>
                <option value="aqua">💧 Aqua</option>
                <option value="lofi">🎶 Lo-fi</option>
                <option value="pastel">🖍️ Pastel</option>
                <option value="fantasy">🧚 Fantasy</option>
                <option value="wireframe">📏 Wireframe</option>
                <option value="black">🖤 Black</option>
                <option value="luxury">💎 Luxury</option>
                <option value="dracula">🧛 Dracula</option>
                <option value="cmyk">🖨️ CMYK</option>
                <option value="autumn">🍂 Autumn</option>
                <option value="business">💼 Business</option>
                <option value="acid">💊 Acid</option>
                <option value="lemonade">🍋 Lemonade</option>
                <option value="night">🌙 Night</option>
                <option value="coffee">☕ Coffee</option>
                <option value="winter">☃️ Winter</option>
                <option value="dim">🌙 Dim</option>
                <option value="nord">🏔️ Nord</option>
                <option value="sunset">🌇 Sunset</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="container mx-auto px-4 sm:px-8 py-8 flex flex-col lg:flex-row gap-8 flex-grow">
        {" "}
        {/* Use flexbox for layout, stack on small screens, row on large, flex-grow to fill height */}
        {/* Left Panel (Input, Simulation, Summary) */}
        <div className="flex flex-col gap-6 lg:w-1/3">
          {" "}
          {/* Allocate 1/3 width on large screens */}
          {error && (
            <div
              role="alert"
              className="alert alert-error w-full shadow-lg mb-4"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="stroke-current shrink-0 h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <span>Error: {error}</span>
            </div>
          )}
          <div className="card w-full bg-base-100 shadow-xl">
            <div className="card-body space-y-4">
              <h2 className="card-title text-base-content">Input</h2>
              <div className="form-control">
                <label className="label">
                  <span className="label-text text-base-content">
                    Regular Expression:
                  </span>
                </label>
                {/* DaisyUI Join for input and button */}
                <div className="join w-full">
                  <input
                    id="regexInput"
                    type="text"
                    value={localRegexInput}
                    onChange={(e) => {
                      // Consider debouncing this input for performance on large regexes
                      setLocalRegexInput(e.target.value);
                      setError(null); // Clear error on input change
                    }}
                    onKeyDown={handleRegexInputKeyDown} // Added specific keydown handler
                    onKeyPress={(e) => {
                      // Prevent default for '+' on keypress as a fallback/alternative
                      if (e.key === "+") {
                        e.preventDefault();
                      }
                    }}
                    placeholder="e.g., a(b|c)*d or (a|b)+c? (\e for epsilon)"
                    className={`join-item input input-bordered flex-grow ${
                      error ? "input-error" : ""
                    }`}
                    aria-label="Regular Expression Input"
                  />
                  <button
                    onClick={handleConvert}
                    disabled={!localRegexInput.trim()}
                    className="join-item btn btn-primary"
                    aria-label="Convert to NFA"
                  >
                    Convert to NFA
                  </button>
                </div>
                {error && (
                  <label className="label">
                    <span className="label-text-alt text-error">{error}</span>
                  </label>
                )}
              </div>
              {/* Regex Examples Dropdown */}
              <div className="form-control">
                <label className="label">
                  <span className="label-text text-base-content">
                    Examples:
                  </span>
                </label>
                <select
                  className="select select-bordered w-full"
                  onChange={(e) => {
                    setLocalRegexInput(e.target.value);
                    setError(null); // Clear error on example selection
                  }}
                  value="" // Placeholder value
                  aria-label="Select Example Regex"
                >
                  <option value="" disabled>
                    Select an example regex
                  </option>
                  {exampleRegexes.map((example, index) => (
                    <option key={index} value={example}>
                      {example === "" ? "Empty String (ε)" : example}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
          <div className="card w-full bg-base-100 shadow-xl">
            <div className="card-body space-y-4">
              <h2 className="card-title text-base-content">Simulation</h2>
              <div className="form-control">
                <label className="label">
                  <span className="label-text text-base-content">
                    Test String:
                  </span>
                </label>
                <input
                  id="testStringInput"
                  type="text"
                  value={testString}
                  onChange={(e) => {
                    setTestString(e.target.value);
                    // Simulation reset is handled by a dedicated useEffect
                  }}
                  placeholder="Enter string to test against NFA"
                  className="input input-bordered w-full"
                  disabled={!nfa} // Disable if no NFA is built
                  aria-label="Test String Input"
                />
              </div>
              {nfa && (
                <>
                  <div className="form-control">
                    <label className="label">
                      <span className="label-text text-base-content">
                        Animation Speed (ms):
                      </span>
                      <span className="label-text-alt">{animationSpeed}ms</span>
                    </label>
                    <input
                      type="range"
                      min="100"
                      max="2000"
                      value={animationSpeed}
                      onChange={(e) =>
                        setAnimationSpeed(Number(e.target.value))
                      }
                      className="range range-primary"
                      step="100"
                      aria-label="Animation Speed Slider"
                    />
                  </div>
                  {/* Speed Preset Buttons */}
                  <div className="flex justify-center gap-4">
                    <button
                      className="btn btn-sm btn-outline"
                      onClick={() => setAnimationSpeed(500)}
                      aria-label="Set Speed to Fast (500ms)"
                    >
                      Fast (500ms)
                    </button>
                    <button
                      className="btn btn-sm btn-outline"
                      onClick={() => setAnimationSpeed(1000)}
                      aria-label="Set Speed to Normal (1000ms)"
                    >
                      Normal (1000ms)
                    </button>
                    <button
                      className="btn btn-sm btn-outline"
                      onClick={() => setAnimationSpeed(2000)}
                      aria-label="Set Speed to Slow (2000ms)"
                    >
                      Slow (2000ms)
                    </button>
                  </div>
                </>
              )}
              {/* Simulation Control Buttons with Icons */}
              <div className="flex flex-col sm:flex-row gap-2 justify-center">
                <button
                  onClick={handleStepBackward}
                  disabled={currentStep === 0 || !nfa || isPlaying} // Disable at step 0 or if no NFA or if playing
                  className="btn btn-circle btn-secondary"
                  aria-label="Step Backward"
                  title="Step Backward (Left Arrow)" // Added tooltip
                >
                  <BackwardIcon className="w-5 h-5" />
                </button>
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  disabled={
                    !testString || !nfa || currentStep >= testString.length // Disable if no test string, no NFA, or at the end
                  }
                  className={`btn btn-circle ${
                    isPlaying ? "btn-warning" : "btn-primary"
                  }`}
                  aria-label={
                    isPlaying ? "Pause Simulation" : "Play Simulation"
                  }
                  title={isPlaying ? "Pause (Space)" : "Play (Space)"} // Added tooltip
                >
                  {isPlaying ? (
                    <PauseIcon className="w-5 h-5" />
                  ) : (
                    <PlayIcon className="w-5 h-5" />
                  )}
                </button>
                <button
                  onClick={handleStepForward}
                  disabled={
                    currentStep >= testString.length || !nfa || isPlaying
                  } // Disable at the end, if no NFA, or if playing
                  className="btn btn-circle btn-secondary"
                  aria-label="Step Forward"
                  title="Step Forward (Right Arrow)" // Added tooltip
                >
                  <ForwardIcon className="w-5 h-5" />
                </button>
              </div>
              {nfa && (
                <div className="text-center text-sm text-base-content pt-2">
                  <span>
                    Processing: "{testString.slice(0, currentStep)}"
                    <span className="font-semibold text-primary">
                      {testString.slice(currentStep, currentStep + 1)}
                    </span>
                    {testString.slice(currentStep + 1)}
                  </span>
                  <br />
                  <span>
                    Step {currentStep} of {testString.length}.{" "}
                  </span>
                  {currentStep === testString.length &&
                    history.length > currentStep && ( // Check if history has the entry for the final step
                      <span
                        className={`ml-1 font-semibold ${
                          isAccepted ? "text-success" : "text-error"
                        }`}
                      >
                        {isAccepted ? "✅ Accepted" : "❌ Rejected"}
                      </span>
                    )}
                </div>
              )}
            </div>
          </div>
          {/* NFA Summary */}
          {nfaSummary && (
            <div className="card w-full bg-base-100 shadow-xl">
              <div className="card-body">
                <h2 className="card-title text-base-content">NFA Details</h2>
                {nfaSummary}
              </div>
            </div>
          )}
        </div>
        {/* Right Panel (Visualization and Guide) */}
        <div className="flex flex-col gap-6 lg:w-2/3">
          {" "}
          {/* Allocate 2/3 width on large screens */}
          <div className="card w-full h-[500px] sm:h-[600px] bg-base-100 shadow-xl border border-base-300 overflow-hidden flex flex-col">
            {" "}
            {/* Use flex-col to make card body fill height */}
            {nfa ? (
              <>
                <div className="card-body p-4 pb-0 flex-row items-center justify-between flex-shrink-0">
                  {" "}
                  {/* Prevent card body from shrinking */}
                  <h2 className="card-title text-base-content">
                    NFA Visualization
                  </h2>
                  <div className="flex gap-2">
                    {/* Combined Reset Layout Button */}
                    <button
                      className="btn btn-sm btn-outline"
                      onClick={handleResetLayout}
                      aria-label="Reset Layout (Stabilize and Fit)"
                      title="Reset Layout (Stabilize and Fit)" // Added tooltip
                    >
                      <ArrowPathIcon className="w-4 h-4 mr-1" />
                      Reset Layout
                    </button>
                  </div>
                </div>
                <div
                  ref={containerRef}
                  className="w-full flex-grow"
                  aria-label="NFA Graph Visualization"
                />
              </>
            ) : (
              <div className="w-full h-full flex items-center justify-center text-base-content text-opacity-70 p-4 text-center">
                <p>
                  Enter a regular expression and click "Convert to NFA" to see
                  the visualization.
                </p>
              </div>
            )}
          </div>
          {/* Guide Section */}
          <div className="card w-full bg-base-100 shadow-xl mb-8">
            <div className="card-body p-0">
              <button
                onClick={() => setIsGuideOpen(!isGuideOpen)}
                className="btn btn-ghost w-full text-left text-base-content flex justify-between items-center rounded-b-none"
                aria-expanded={isGuideOpen}
                aria-controls="guide-content"
              >
                <span className="font-semibold flex items-center">
                  <QuestionMarkCircleIcon className="w-5 h-5 mr-2" />
                  How to Use This Tool
                </span>
                <svg
                  className={`w-5 h-5 transition-transform ${
                    isGuideOpen ? "transform rotate-180" : ""
                  }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </button>
              {isGuideOpen && (
                <div
                  id="guide-content"
                  className="p-4 space-y-3 text-sm text-base-content text-opacity-80 border-t border-base-200"
                >
                  <p>
                    1. Enter a regular expression in the input field (e.g.,{" "}
                    <code>a(b|c)*d</code>, <code>(x|y)+</code>, use{" "}
                    <code>\e</code> for an explicit epsilon transition).
                    Supported operators: <code>|</code> (alternation), implicit
                    and explicit <code>·</code> (concatenation), <code>*</code>{" "}
                    (Kleene star), <code>+</code> (Kleene plus), <code>?</code>{" "}
                    (zero or one). Parentheses <code>()</code> for grouping. You
                    can escape metacharacters like <code>\*</code>,{" "}
                    <code>\+</code>, etc., using a backslash <code>\</code>.
                  </p>
                  <ul className="list-disc pl-5 mt-1 space-y-1">
                    <li>
                      Start state is labeled 'Start' with a{" "}
                      <span className="text-blue-600 font-semibold">blue</span>{" "}
                      background and thicker border.
                    </li>
                    <li>
                      Accept states are labeled 'Accept' and colored{" "}
                      <span className="text-green-600 font-semibold">
                        green
                      </span>
                      .
                    </li>
                    <li>
                      Normal states are labeled with their ID (e.g., '0', '1')
                      with a{" "}
                      <span className="text-gray-700 font-semibold">gray</span>{" "}
                      background.
                    </li>
                    <li>
                      Active states during simulation have a{" "}
                      <span className="text-yellow-500 font-semibold">
                        yellow background and subtle pulse effect
                      </span>
                      .
                    </li>
                    <li>
                      Epsilon (ε) transitions are shown as dashed, curved lines,
                      often purple. Other transitions are straight lines with
                      varying colors based on the symbol.
                    </li>
                    <li>
                      Hover over nodes or edges for more details (ID, symbol)
                      via tooltips.
                    </li>
                  </ul>
                  <p className="mt-3">
                    2. Enter a string in the <strong>Test String</strong> field
                    to simulate its processing by the NFA. Use the{" "}
                    <strong>Step Forward</strong> (→),{" "}
                    <strong>Step Back</strong> (←), and{" "}
                    <strong>Play/Pause</strong> buttons to control the
                    simulation. You can also use the Left/Right arrow keys to
                    step and Spacebar to play/pause (when input fields are not
                    focused). Adjust the Animation Speed slider or use the
                    preset buttons to change the step delay.
                  </p>
                  <p>
                    3. Observe the highlighted states in the NFA graph. These
                    are the states the NFA could be in after processing the
                    characters up to the current step.
                  </p>
                  <p>
                    4. After processing the entire string (Step equals the
                    string length), the result (Accepted or Rejected) will be
                    shown. The string is accepted if at least one of the
                    highlighted states is an Accept state ('Accept').
                  </p>
                  <p>
                    5. Use the "Reset Layout" button to re-arrange and fit the
                    graph to the screen.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      {/* Global Footer */}
      <footer className="footer footer-center p-4 bg-base-300 text-base-content">
        <aside>
          <p>
            Built with React, vis-network, and DaisyUI. | © 2023 Your
            Name/Project Name
          </p>
        </aside>
      </footer>
    </div>
  );
};

export default App;
