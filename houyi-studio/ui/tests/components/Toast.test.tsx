import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { ToastContainer } from '@/components/Toast';

describe('ToastContainer', () => {
  it('renders above modal overlay layer', () => {
    render(
      <ToastContainer
        toasts={[{ id: '1', message: 'Skill loaded', type: 'success' }]}
        onRemove={vi.fn()}
      />,
    );

    expect(screen.getByTestId('toast-container')).toHaveClass('z-[70]');
  });
});
