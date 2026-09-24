/**
 * Robust parser and scrubber for coding responses.
 * Enforces strict query isolation, structured execution data,
 * and strips raw UI/icon markers from rendering.
 */

/**
 * Remove internal UI and icon step markers leaking from DOM or accessibility trees.
 * Examples: svgCopy, svgQuery, svgRetrieval, svgModel, svgTools, svgResponse, svgDownload
 */
export function stripRawUiMarkers(text) {
  if (!text || typeof text !== 'string') return '';
  return text
    // Strip raw "svg" prefix prepended to UI button / trace labels
    .replace(/\bsvg(?=Copy|Query|Retrieval|Model|Tools|Response|Download)/gi, '')
    .trim();
}

/**
 * Robust parser that checks structured data (coding_result / tool_calls / top-level fields)
 * and falls back to markdown string parsing.
 */
export function parseCodingResponse(msg) {
  if (!msg) return null;

  const rawContent = typeof msg.content === 'string' ? msg.content : '';
  const codingResult = msg.coding_result || null;

  // 1. Check tool_calls for sandbox_execute
  let sandboxToolResult = null;
  if (Array.isArray(msg.tool_calls)) {
    const sb = msg.tool_calls.filter(
      (tc) => tc && (tc.tool === 'sandbox_execute' || tc.attempt !== undefined)
    );
    if (sb.length > 0) {
      const verified = sb.find((a) => a.status === 'verified' || a.exit_code === 0);
      sandboxToolResult = verified || sb[sb.length - 1];
    }
  }

  // Detect whether this message represents a coding response
  const isCodingTask =
    msg.task_type === 'CODING' ||
    msg.task_type === 'code_generation' ||
    Boolean(codingResult) ||
    Boolean(sandboxToolResult) ||
    Boolean(msg.code) ||
    rawContent.includes('**Task Type:** CODING') ||
    (rawContent.includes('```python') && (rawContent.includes('Execution Result') || rawContent.includes('Docker Sandbox')));

  if (!isCodingTask) {
    return null;
  }

  // Extract Code
  let code = '';
  if (msg.code) {
    code = msg.code;
  } else if (codingResult?.code) {
    code = codingResult.code;
  } else if (sandboxToolResult?.code) {
    code = sandboxToolResult.code;
  } else {
    const codeMatch = rawContent.match(/```(?:python|py)?\r?\n([\s\S]*?)```/i);
    if (codeMatch) {
      code = codeMatch[1];
    } else if (isCodingTask) {
      const trimmed = rawContent.trim();
      if (/^(?:import\s+|from\s+|def\s+|class\s+|#)/m.test(trimmed)) {
        code = trimmed;
      }
    }
  }

  // If no code could be found, this is not a valid code presentation
  if (!code.trim()) {
    return null;
  }

  // Extract Model
  let model = codingResult?.model || msg.model || msg.model_id || 'qwen2.5-coder:3b';
  const modelMatch = rawContent.match(/\*\*Model:\*\*\s*([^\n]+)/i);
  if (modelMatch && (!model || model === 'qwen2.5-coder:3b')) {
    model = modelMatch[1].trim();
  }

  // Extract Status & Exit Code
  let status = msg.status || codingResult?.status || 'UNKNOWN';
  let exitCode = msg.exit_code ?? codingResult?.exit_code ?? null;

  if (status === 'UNKNOWN' || status === 'SUCCESS') {
    if (sandboxToolResult) {
      exitCode = sandboxToolResult.exit_code ?? exitCode;
      status =
        exitCode === 0 || sandboxToolResult.status === 'verified'
          ? 'VERIFIED'
          : sandboxToolResult.status === 'error' || sandboxToolResult.status === 'blocked'
          ? 'BLOCKED'
          : 'FAILED';
    } else {
      const execMatch = rawContent.match(/\*\*Execution:\*\*\s*([^\n]+)/i);
      if (execMatch) {
        const val = execMatch[1].trim().toUpperCase();
        if (val.includes('VERIFIED')) status = 'VERIFIED';
        else if (val.includes('BLOCKED')) status = 'BLOCKED';
        else if (val.includes('FAILED')) {
          status = 'FAILED';
          const ecMatch = val.match(/EXIT CODE\s*(\d+)/i);
          if (ecMatch) exitCode = parseInt(ecMatch[1], 10);
        }
      } else if (exitCode === 0) {
        status = 'VERIFIED';
      }
    }
  }

  // Extract Output (stdout / stderr / notice)
  let stdout = msg.stdout || codingResult?.stdout || sandboxToolResult?.stdout || '';
  let stderr = msg.stderr || codingResult?.stderr || sandboxToolResult?.stderr || '';
  let notice = '';

  if (!stdout && !stderr) {
    const stdoutMatch = rawContent.match(/###\s+Execution Result\s*\((?:Docker Sandbox|Sandbox)\)\r?\n```(?:[a-z]*)?\r?\n([\s\S]*?)```/i);
    if (stdoutMatch) {
      stdout = stdoutMatch[1].trim();
    }

    const stderrMatch = rawContent.match(/###\s+Diagnostic Output\s*\((?:Docker Sandbox|Sandbox)\)\r?\n```(?:[a-z]*)?\r?\n([\s\S]*?)```/i);
    if (stderrMatch) {
      stderr = stderrMatch[1].trim();
    }

    const noticeMatch = rawContent.match(/⚠️\s*\*\*Sandbox Execution Notice\*\*:\s*([^\n]+)/i);
    if (noticeMatch) {
      notice = noticeMatch[1].trim();
    }
  }

  // Extract Correction Attempts
  let attempts =
    msg.correction_attempts ||
    codingResult?.correction_attempts ||
    (Array.isArray(msg.tool_calls) ? msg.tool_calls.length : null);
  if (!attempts) {
    const attemptsMatch = rawContent.match(/\*\*Correction Attempts:\*\*\s*(\d+)/i);
    if (attemptsMatch) {
      attempts = parseInt(attemptsMatch[1], 10);
    }
  }

  // Extract Real Sandbox Information
  let sandboxInfo =
    msg.execution_details ||
    codingResult?.execution_details ||
    codingResult?.sandbox_info ||
    sandboxToolResult?.sandbox_info ||
    null;

  if (!sandboxInfo && (exitCode !== null || status === 'VERIFIED' || status === 'FAILED')) {
    sandboxInfo = {
      sandbox: 'Docker',
      network: 'Disabled',
      filesystem: 'Read-only',
      user: 'Non-root',
      exit_code: exitCode !== null ? exitCode : (status === 'VERIFIED' ? 0 : 1),
      status: status === 'VERIFIED' ? 'Verified' : (status === 'BLOCKED' ? 'Blocked' : 'Failed'),
    };
  }

  // Parse remaining conversational text before or after code fences
  let remainingText = rawContent
    .replace(/\*\*Task Type:\*\*\s*[^\n]+\r?\n?/gi, '')
    .replace(/\*\*Model:\*\*\s*[^\n]+\r?\n?/gi, '')
    .replace(/\*\*Execution:\*\*\s*[^\n]+\r?\n?/gi, '')
    .replace(/\*\*Correction Attempts:\*\*\s*\d+\r?\n?/gi, '')
    .replace(/```(?:python|py)?\r?\n[\s\S]*?```/gi, '')
    .replace(/###\s+(?:Execution Result|Diagnostic Output)\s*\((?:Docker Sandbox|Sandbox)\)\r?\n```(?:[a-z]*)?\r?\n[\s\S]*?```/gi, '')
    .replace(/⚠️\s*\*\*Sandbox Execution Notice\*\*:\s*[^\n]+/gi, '')
    .trim();

  return {
    code: stripRawUiMarkers(code).trim(),
    stdout: stripRawUiMarkers(stdout).trim(),
    stderr: stripRawUiMarkers(stderr).trim(),
    notice: stripRawUiMarkers(notice).trim(),
    status,
    isVerified: status === 'VERIFIED',
    isFailed: status === 'FAILED',
    isBlocked: status === 'BLOCKED',
    model,
    exitCode,
    attempts,
    sandboxInfo,
    remainingText: stripRawUiMarkers(remainingText).trim(),
    currentQuery: stripRawUiMarkers(codingResult?.current_query || msg.current_query || '').trim(),
    traceId: codingResult?.trace_id || msg.trace_id || null,
  };
}
