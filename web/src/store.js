// 跨视图共享的会话状态：让「会话管理」能跳到「对话」并载入指定历史会话。
import { ref } from 'vue'

export const activeTab = ref('chat')

// Chat 需要载入的历史会话；为 null 表示无待处理切换
export const pendingSessionId = ref(null)

export function openSession(sessionId) {
  pendingSessionId.value = sessionId
  activeTab.value = 'chat'
}

export function clearPending() {
  pendingSessionId.value = null
}