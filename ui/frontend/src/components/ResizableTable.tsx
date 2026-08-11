import { Table as AntTable, type TableProps } from 'antd';
import type { ColumnGroupType, ColumnType, ColumnsType } from 'antd/es/table';
import { useCallback, useMemo, useState, type PointerEvent, type ThHTMLAttributes } from 'react';

const DEFAULT_COLUMN_WIDTH = 180;
const MIN_COLUMN_WIDTH = 96;

type ResizableHeaderCellProps = ThHTMLAttributes<HTMLTableCellElement> & {
  columnKey?: string;
  onColumnsResize?: (widths: Record<string, number>) => void;
};

function redistributeColumnWidths(
  widths: Record<string, number>,
  targetKey: string,
  requestedDelta: number,
) {
  const entries = Object.entries(widths);
  const targetWidth = widths[targetKey];
  const otherEntries = entries.filter(([key]) => key !== targetKey);
  if (targetWidth == null || !otherEntries.length) return widths;

  const minimum = (width: number) => Math.min(MIN_COLUMN_WIDTH, width);
  let delta = requestedDelta;

  if (delta > 0) {
    const shrinkCapacity = otherEntries.reduce(
      (total, [, width]) => total + Math.max(0, width - minimum(width)),
      0,
    );
    delta = Math.min(delta, shrinkCapacity);
  } else {
    delta = Math.max(delta, minimum(targetWidth) - targetWidth);
  }

  const result: Record<string, number> = {
    ...widths,
    [targetKey]: targetWidth + delta,
  };

  if (delta > 0) {
    const totalCapacity = otherEntries.reduce(
      (total, [, width]) => total + Math.max(0, width - minimum(width)),
      0,
    );
    otherEntries.forEach(([key, width]) => {
      const capacity = Math.max(0, width - minimum(width));
      result[key] = width - delta * (capacity / totalCapacity);
    });
  } else if (delta < 0) {
    const releasedWidth = -delta;
    const totalOtherWidth = otherEntries.reduce((total, [, width]) => total + width, 0);
    otherEntries.forEach(([key, width]) => {
      result[key] = width + releasedWidth * (width / totalOtherWidth);
    });
  }

  const rounded = Object.fromEntries(
    Object.entries(result).map(([key, width]) => [key, Math.round(width)]),
  );
  const originalTotal = Math.round(entries.reduce((total, [, width]) => total + width, 0));
  const roundedTotal = Object.values(rounded).reduce((total, width) => total + width, 0);
  const correctionKey = otherEntries.at(-1)?.[0] ?? targetKey;
  rounded[correctionKey] += originalTotal - roundedTotal;
  return rounded;
}

function ResizableHeaderCell({
  children,
  columnKey,
  onColumnsResize,
  ...cellProps
}: ResizableHeaderCellProps) {
  const startResize = (event: PointerEvent<HTMLSpanElement>) => {
    if (!columnKey || !onColumnsResize) return;

    event.preventDefault();
    event.stopPropagation();

    const headerCell = event.currentTarget.parentElement;
    const startX = event.clientX;
    const headerCells = Array.from(
      headerCell?.parentElement?.querySelectorAll<HTMLTableCellElement>('th[data-column-key]') ?? [],
    );
    const startWidths = Object.fromEntries(headerCells.map((cell) => [
      cell.dataset.columnKey!,
      cell.getBoundingClientRect().width || DEFAULT_COLUMN_WIDTH,
    ]));

    document.documentElement.classList.add('table-column-resizing');

    const handlePointerMove = (moveEvent: globalThis.PointerEvent) => {
      onColumnsResize(redistributeColumnWidths(
        startWidths,
        columnKey,
        moveEvent.clientX - startX,
      ));
    };
    const stopResize = () => {
      document.documentElement.classList.remove('table-column-resizing');
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', stopResize);
      window.removeEventListener('pointercancel', stopResize);
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', stopResize);
    window.addEventListener('pointercancel', stopResize);
  };

  return (
    <th {...cellProps} data-column-key={columnKey}>
      {children}
      {columnKey && onColumnsResize ? (
        <span
          aria-hidden="true"
          className="table-column-resize-handle"
          onPointerDown={startResize}
        />
      ) : null}
    </th>
  );
}

function columnIdentifier<RecordType>(
  column: ColumnType<RecordType> | ColumnGroupType<RecordType>,
  path: string,
) {
  if (column.key != null) return String(column.key);
  if ('dataIndex' in column && column.dataIndex != null) {
    return Array.isArray(column.dataIndex) ? column.dataIndex.join('.') : String(column.dataIndex);
  }
  return `column-${path}`;
}

export function ResizableTable<RecordType extends object = Record<string, unknown>>({
  columns,
  components,
  scroll,
  className,
  ...tableProps
}: TableProps<RecordType>) {
  const [columnWidths, setColumnWidths] = useState<Record<string, number>>({});

  const resizeColumns = useCallback((widths: Record<string, number>) => {
    setColumnWidths(widths);
  }, []);

  const resizableColumns = useMemo(() => {
    const enhance = (
      source: ColumnsType<RecordType>,
      parentPath = '',
    ): ColumnsType<RecordType> => source.map((column, index) => {
      const path = parentPath ? `${parentPath}-${index}` : String(index);
      const key = columnIdentifier(column, path);

      if ('children' in column && column.children) {
        return {
          ...column,
          children: enhance(column.children, path),
        };
      }

      const leaf = column as ColumnType<RecordType>;
      return {
        ...leaf,
        fixed: undefined,
        width: columnWidths[key] ?? (
          typeof leaf.width === 'number' && leaf.width <= MIN_COLUMN_WIDTH ? leaf.width : undefined
        ),
        onHeaderCell: (headerColumn) => ({
          ...(leaf.onHeaderCell?.(headerColumn) ?? {}),
          columnKey: key,
          onColumnsResize: resizeColumns,
        }),
      };
    });

    return columns ? enhance(columns) : columns;
  }, [columns, columnWidths, resizeColumns]);

  const fittedScroll = scroll ? { ...scroll, x: undefined } : undefined;

  return (
    <AntTable<RecordType>
      {...tableProps}
      className={['resizable-table', className].filter(Boolean).join(' ')}
      columns={resizableColumns}
      tableLayout={tableProps.tableLayout ?? 'fixed'}
      components={{
        ...components,
        header: {
          ...components?.header,
          cell: ResizableHeaderCell,
        },
      }}
      scroll={fittedScroll}
    />
  );
}
