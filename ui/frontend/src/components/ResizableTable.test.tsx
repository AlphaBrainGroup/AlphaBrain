/** @vitest-environment jsdom */

import { cleanup, fireEvent, render } from '@testing-library/react';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { ResizableTable } from './ResizableTable';

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', class {
    observe() {}
    unobserve() {}
    disconnect() {}
  });
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });
});

afterEach(cleanup);

describe('ResizableTable', () => {
  it('fits the panel initially even when a page requests a wider scrolling table', () => {
    const { container } = render(
      <ResizableTable
        pagination={false}
        rowKey="id"
        scroll={{ x: 1300 }}
        dataSource={[{ id: 'row-1', workload: 'Training run', actions: 'View' }]}
        columns={[
          { title: 'Workload', dataIndex: 'workload', fixed: 'left' },
          { title: 'Status', render: () => 'Running' },
          { title: 'Owner', render: () => 'Owner' },
          { title: 'GPU', render: () => 'GPU 0' },
          { title: 'Package', width: 290, render: () => 'Package controls' },
          { title: 'Actions', dataIndex: 'actions', fixed: 'right' },
        ]}
      />,
    );

    expect(container.querySelector('table')?.style.width).toBe('');
    expect(container.querySelector('.ant-table-cell-fix-left')).toBeNull();
    expect(container.querySelector('.ant-table-cell-fix-right')).toBeNull();
    expect(container.querySelector('col[style*="290px"]')).toBeNull();
    expect(container.querySelector('.resizable-table--resized')).toBeNull();
  });

  it('resizes a column by compressing the others without changing total table width', () => {
    const { container } = render(
      <ResizableTable
        pagination={false}
        rowKey="id"
        dataSource={[{ id: 'row-1', name: 'A long value that should wrap inside the cell' }]}
        columns={[
          { title: 'ID', dataIndex: 'id' },
          { title: 'Name', dataIndex: 'name' },
        ]}
      />,
    );

    const handles = container.querySelectorAll<HTMLElement>('.table-column-resize-handle');
    expect(handles).toHaveLength(2);
    expect(container.querySelector('table')?.style.width).toBe('');
    expect(container.querySelector('col')).toBeNull();

    const firstHeader = handles[0].parentElement as HTMLTableCellElement;
    firstHeader.getBoundingClientRect = () => ({
      width: 180,
      height: 40,
      top: 0,
      right: 180,
      bottom: 40,
      left: 0,
      x: 0,
      y: 0,
      toJSON: () => undefined,
    });
    const secondHeader = handles[1].parentElement as HTMLTableCellElement;
    secondHeader.getBoundingClientRect = () => ({
      width: 180,
      height: 40,
      top: 0,
      right: 360,
      bottom: 40,
      left: 180,
      x: 180,
      y: 0,
      toJSON: () => undefined,
    });

    fireEvent.pointerDown(handles[0], { clientX: 180 });
    fireEvent.pointerMove(window, { clientX: 260 });
    fireEvent.pointerUp(window);

    const resizedColumns = Array.from(container.querySelectorAll('col')).map((column) => (
      Number.parseInt(column.style.width, 10)
    ));
    expect(resizedColumns).toEqual([260, 100]);
    expect(resizedColumns.reduce((total, width) => total + width, 0)).toBe(360);
    expect(container.querySelector('table')?.style.width).toBe('');
    expect(container.querySelector('.resizable-table--resized')).toBeNull();
  });
});
