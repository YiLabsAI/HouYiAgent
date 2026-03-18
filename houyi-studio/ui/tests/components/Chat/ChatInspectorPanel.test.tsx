import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ChatInspectorPanel } from '@/components/Chat/ChatInspectorPanel';

describe('ChatInspectorPanel', () => {
  it('shows snapshot data without trace stage names', () => {
    render(
      <ChatInspectorPanel
        conversationContext={{
          conversation_id: 'conv-1',
          used_units: 820,
          max_units: 1000,
          state: 'near_compaction',
          last_compacted_at: 1710000000,
          last_compaction_delta: 220,
          updated_at: 1710000100,
        }}
        contextUsage={{
          model: 'deepseek-chat',
          max_context_tokens: 8192,
          used_tokens: 1520,
          reserved_output_tokens: 1024,
          available_tokens: 5648,
          available_input_tokens: 5648,
          planned_prompt_tokens: 1520,
          block_breakdown: {
            recent: 1200,
            pinned: 320,
          },
          drop_reasons: {
            'dropped-memory': 'boundary_excluded',
          },
          dropped_block_details: [
            {
              candidate_id: 'dropped-memory',
              block_type: 'memory',
              source: 'memory',
              token_count: 42,
              message_count: 2,
              pinned: false,
            },
          ],
          timestamp: 1710000100,
        }}
        latestCompaction={{
          compaction_id: 'cmp-1',
          trigger: 'pre_request_pressure',
          summary: 'Compacted old history',
          source_message_ids: ['m1'],
          pinned_message_ids: [],
          retained_refs: [],
          metrics: {
            messages_compacted: 4,
            tokens_before: 4800,
            tokens_after: 2600,
            pin_violation_count: 0,
          },
          created_at: 1710000100,
          metadata: {},
        }}
        activePins={[]}
        usage={{
          prompt_tokens: 120,
          completion_tokens: 40,
          cached_prompt_tokens: 12,
          cached_prompt_tokens_reported: true,
          cache_hit: true,
          cache_hit_reported: true,
          usage_confidence: 'high',
        }}
        onUpdatePinnedContextStatus={vi.fn()}
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByText('Chat Inspector')).toBeInTheDocument();
    expect(screen.getByText('Compacted conversation context for this request')).toBeInTheDocument();
    expect(screen.getByText('Recent')).toBeInTheDocument();
    expect(screen.getByText('Pinned')).toBeInTheDocument();
    expect(screen.getByText('Pins protected')).toBeInTheDocument();
    expect(screen.getByText('Yes')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Show trimmed details' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Show trimmed details' }));
    expect(screen.getByText('Memory')).toBeInTheDocument();
    expect(screen.getByText(/2 msgs · 42 tokens/i)).toBeInTheDocument();
    expect(screen.queryByText(/Internal reference: memory/i)).not.toBeInTheDocument();
    expect(screen.queryByText('context.compaction')).not.toBeInTheDocument();
  });

  it('renders human-readable labels for additional trim reasons', () => {
    render(
      <ChatInspectorPanel
        conversationContext={null}
        contextUsage={{
          model: 'deepseek-chat',
          max_context_tokens: 8192,
          used_tokens: 1520,
          reserved_output_tokens: 1024,
          available_tokens: 5648,
          available_input_tokens: 5648,
          planned_prompt_tokens: 1520,
          block_breakdown: {
            current_turn: 320,
          },
          drop_reasons: {
            recent_a: 'policy_excluded',
            recent_b: 'excluded_without_current_turn',
          },
          timestamp: 1710000200,
        }}
        latestCompaction={null}
        activePins={[]}
        usage={null}
        onUpdatePinnedContextStatus={vi.fn()}
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByText('Excluded by request policy')).toBeInTheDocument();
    expect(screen.getByText('Excluded because the current turn could not be kept')).toBeInTheDocument();
  });

  it('shows token accounting when only cache-hit observability is reported', () => {
    render(
      <ChatInspectorPanel
        conversationContext={null}
        contextUsage={null}
        latestCompaction={null}
        activePins={[]}
        usage={{
          cache_hit: false,
          cache_hit_reported: true,
        }}
        onUpdatePinnedContextStatus={vi.fn()}
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByText('Token Accounting')).toBeInTheDocument();
    expect(screen.getByText('Cache hit')).toBeInTheDocument();
    expect(screen.getByText('No')).toBeInTheDocument();
    expect(screen.getByText(/Derived from provider-reported cache usage fields/i)).toBeInTheDocument();
  });

  it('prefers metadata timing metrics over usage fallback values', () => {
    render(
      <ChatInspectorPanel
        conversationContext={null}
        contextUsage={null}
        latestCompaction={null}
        activePins={[]}
        usage={{
          first_token_ms: 321,
          first_token_latency_ms: 187,
          generation_time_ms: 912,
          decode_time_ms: 725,
          decode_tokens_per_second: 27,
          end_to_end_tokens_per_second: 22,
          tokens_per_second: 19,
        }}
        onUpdatePinnedContextStatus={vi.fn()}
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByText('187.00 ms')).toBeInTheDocument();
    expect(screen.getByText('912.00 ms')).toBeInTheDocument();
    expect(screen.getByText('725.00 ms')).toBeInTheDocument();
    expect(screen.getByText('27.00 tok/s')).toBeInTheDocument();
    expect(screen.getByText('22.00 tok/s')).toBeInTheDocument();
    expect(screen.queryByText('321.00 ms')).not.toBeInTheDocument();
    expect(screen.queryByText('19.00 tok/s')).not.toBeInTheDocument();
  });

  it('shows token accounting when only local timing metadata exists', () => {
    render(
      <ChatInspectorPanel
        conversationContext={null}
        contextUsage={null}
        latestCompaction={null}
        activePins={[]}
        usage={{
          first_token_latency_ms: 95,
        }}
        onUpdatePinnedContextStatus={vi.fn()}
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByText('Token Accounting')).toBeInTheDocument();
    expect(screen.getByText('95.00 ms')).toBeInTheDocument();
  });

  it('shows trimmed details when only dropped block details are reported', () => {
    render(
      <ChatInspectorPanel
        conversationContext={null}
        contextUsage={{
          model: 'deepseek-chat',
          max_context_tokens: 8192,
          used_tokens: 1520,
          reserved_output_tokens: 1024,
          available_tokens: 5648,
          available_input_tokens: 5648,
          planned_prompt_tokens: 1520,
          block_breakdown: {
            recent: 1200,
          },
          drop_reasons: {},
          dropped_block_details: [
            {
              candidate_id: 'memory-block',
              block_type: 'memory',
              source: 'memory',
              token_count: 42,
              message_count: 2,
              pinned: false,
            },
          ],
          timestamp: 1710000300,
        }}
        latestCompaction={null}
        activePins={[]}
        usage={null}
        onUpdatePinnedContextStatus={vi.fn()}
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByText('Context trimmed')).toBeInTheDocument();
    expect(screen.getByText('1')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Show trimmed details' }));
    expect(screen.getByText('Memory')).toBeInTheDocument();
    expect(screen.getByText('Omitted from request context')).toBeInTheDocument();
    expect(screen.getByText(/2 msgs · 42 tokens/i)).toBeInTheDocument();
  });

  it('updates pin status from active pins actions', () => {
    const onUpdatePinnedContextStatus = vi.fn();

    render(
      <ChatInspectorPanel
        conversationContext={null}
        contextUsage={null}
        latestCompaction={null}
        activePins={[
          {
            pin_id: 'pin-1',
            conversation_id: 'conv-1',
            source_message_id: 'u1',
            title: 'Constraint',
            content: 'Deploy to staging first.',
            role: 'user',
            scope: 'conversation',
            status: 'active',
            priority: 5,
            token_count: 8,
            created_at: 1710000000,
            updated_at: 1710000100,
            metadata: {},
          },
        ]}
        usage={null}
        onUpdatePinnedContextStatus={onUpdatePinnedContextStatus}
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByText('Role')).toBeInTheDocument();
    expect(screen.getByText('user')).toBeInTheDocument();
    expect(screen.getByText('Tokens')).toBeInTheDocument();
    expect(screen.getByText('8')).toBeInTheDocument();
    expect(screen.getByText('Updated')).toBeInTheDocument();
    expect(screen.getByText(/Add or replace pins from a message's pin menu/i)).toBeInTheDocument();
    expect(screen.queryByText('Supersede')).not.toBeInTheDocument();
    fireEvent.click(screen.getByText('Unpin'));
    fireEvent.click(screen.getByText('Archive'));

    expect(onUpdatePinnedContextStatus).toHaveBeenNthCalledWith(1, 'pin-1', 'removed');
    expect(onUpdatePinnedContextStatus).toHaveBeenNthCalledWith(2, 'pin-1', 'archived');
  });

  it('confirms before restoring a snapshot', () => {
    const onRestoreCompaction = vi.fn();

    render(
      <ChatInspectorPanel
        conversationContext={null}
        contextUsage={null}
        latestCompaction={{
          compaction_id: 'cmp-1',
          trigger: 'manual',
          summary: 'Compacted old history',
          source_message_ids: ['u1'],
          pinned_message_ids: [],
          retained_refs: [],
          metrics: {
            messages_compacted: 2,
            tokens_before: 2400,
            tokens_after: 1200,
          },
          created_at: 1710000100,
          metadata: {},
        }}
        compactionHistory={[
          {
            compaction: {
              compaction_id: 'cmp-1',
              trigger: 'manual',
              summary: 'Compacted old history',
              source_message_ids: ['u1'],
              pinned_message_ids: [],
              retained_refs: [],
              metrics: {
                messages_compacted: 2,
                tokens_before: 2400,
                tokens_after: 1200,
              },
              created_at: 1710000100,
              metadata: {},
            },
            diff: {
              source_message_ids: ['u1'],
              backup_message_count: 4,
              current_message_count: 2,
              backup_visible_message_count: 4,
              current_visible_message_count: 2,
              removed_message_ids: ['u1'],
              added_message_ids: [],
            },
            backup: {
              backup_id: 'backup-1',
              conversation_id: 'conv-1',
              trigger: 'restore_point',
              created_at: 1710000100,
              path: 'conv-1--backup-1.json',
              record_id: 'cmp-1',
              metadata: {
                kind: 'restore_point',
                reason: 'before_restore_compaction',
                source_backup_id: 'backup-source-1',
                restored_compaction_id: 'cmp-0',
              },
            },
          },
        ]}
        activePins={[]}
        usage={null}
        onRestoreCompaction={onRestoreCompaction}
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByText('1 snapshot')).toBeInTheDocument();
    expect(screen.queryByText('Snapshot backup-1')).not.toBeInTheDocument();
    expect(screen.queryByText('Restores cmp-0')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Show history' }));
    expect(screen.getByTestId('compaction-history-list')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Restore…' }));
    expect(screen.getByText('Latest compaction details')).toBeInTheDocument();
    expect(screen.getByText('Snapshot')).toBeInTheDocument();
    expect(screen.getByTestId('compaction-added-preview').querySelector('button')).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Show advanced' }));
    expect(screen.getByText('Snapshot trigger')).toBeInTheDocument();
    expect(screen.getAllByText('restore_point').length).toBeGreaterThan(0);
    expect(screen.getByText('Snapshot kind')).toBeInTheDocument();
    expect(screen.getByText('Snapshot reason')).toBeInTheDocument();
    expect(screen.getByText('before_restore_compaction')).toBeInTheDocument();
    expect(screen.getByText('Source backup')).toBeInTheDocument();
    expect(screen.getByText('backup-source-1')).toBeInTheDocument();
    expect(screen.getByText('Restored compaction')).toBeInTheDocument();
    expect(screen.getByText('cmp-0')).toBeInTheDocument();
    expect(screen.getByText(/rollback snapshot created immediately before applying a restore action/i)).toBeInTheDocument();
    expect(screen.getByText(/Total message count now versus the message count stored inside the selected snapshot backup/i)).toBeInTheDocument();
    expect(screen.getByText(/Visible non-system messages now versus the visible message count inside the selected snapshot backup/i)).toBeInTheDocument();
    expect(screen.getByTestId('restore-confirmation-dialog')).toBeInTheDocument();
    expect(screen.getByText('Snapshot ID: backup-1')).toBeInTheDocument();
    expect(screen.getByText('backup-1')).toBeInTheDocument();
    expect(onRestoreCompaction).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'Restore snapshot' }));
    expect(onRestoreCompaction).toHaveBeenCalledWith('cmp-1');
  });

  it('shows rollback affordance inside inspector restore status', () => {
    const onRestoreBackup = vi.fn();
    const onClearRestoreNotice = vi.fn();

    render(
      <ChatInspectorPanel
        conversationContext={null}
        contextUsage={null}
        latestCompaction={null}
        compactionHistory={[]}
        restoringBackupId={null}
        restoreNotice={{
          message: 'Restored snapshot. You can undo this restore from the previous state backup.',
          undoBackupId: 'backup-undo-1',
        }}
        activePins={[]}
        usage={null}
        onRestoreBackup={onRestoreBackup}
        onClearRestoreNotice={onClearRestoreNotice}
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByText('Restore status')).toBeInTheDocument();
    expect(screen.getByText('Restore-point backup')).toBeInTheDocument();
    expect(screen.getByText('backup-undo-1')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Undo restore' }));
    expect(onRestoreBackup).toHaveBeenCalledWith('backup-undo-1');
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }));
    expect(onClearRestoreNotice).toHaveBeenCalled();
  });
});
