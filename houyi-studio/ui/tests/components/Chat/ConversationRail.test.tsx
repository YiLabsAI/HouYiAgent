import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ConversationRail } from '@/components/Chat/ConversationRail';
import type { ConversationSummary } from '@/types/chat';

const conversations: ConversationSummary[] = [
  {
    conversation_id: 'conv_1',
    title: 'Long Chat',
    status: 'active',
    message_count: 209,
    visible_message_count: 148,
    model: 'gpt-4o',
    created_at: 1700000000,
    updated_at: 1700000100,
    last_message_at: 1700000100,
    bookmarked: false,
  },
];

describe('ConversationRail', () => {
  it('uses visible_message_count as the primary sidebar total when available', () => {
    render(
      <ConversationRail
        conversations={conversations}
        activeConversationId="conv_1"
        isLoading={false}
        onSelect={vi.fn()}
        onCreate={vi.fn()}
        onDelete={vi.fn()}
        onOpenSettings={vi.fn()}
        onToggleBookmark={vi.fn()}
      />,
    );

    expect(screen.getByText('148 messages')).toBeInTheDocument();
    expect(screen.queryByText('209 messages')).not.toBeInTheDocument();
  });
});
