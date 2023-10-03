

# import logic_gate class
import logic_gate as lg

logic_gate = lg.LogicGate()

# testing AND gate 
logic_gate.and_gate(1, 0)
logic_gate.print_output('AND')

logic_gate.and_gate(1, 1)
logic_gate.print_output('AND')

logic_gate.and_gate(0, 1)
logic_gate.print_output('AND')

logic_gate.and_gate(0, 0)
logic_gate.print_output('AND')

# testing or gate 
logic_gate.or_gate(1, 0)
logic_gate.print_output('OR')

logic_gate.or_gate(0, 1)
logic_gate.print_output('OR')

logic_gate.or_gate(1, 1)
logic_gate.print_output('OR')

logic_gate.or_gate(0, 0)
logic_gate.print_output('OR')


# testing NAND gate 
logic_gate.nand_gate(1, 0)
logic_gate.print_output('NAND')

logic_gate.nand_gate(0, 1)
logic_gate.print_output('NAND')

logic_gate.nand_gate(1, 1)
logic_gate.print_output('NAND')

logic_gate.nand_gate(0, 0)
logic_gate.print_output('NAND')

# testing NOR gate
logic_gate.nor_gate(1, 0)
logic_gate.print_output('NOR')

logic_gate.nor_gate(0,1)
logic_gate.print_output('NOR')

logic_gate.nor_gate(1, 1)
logic_gate.print_output('NOR')

logic_gate.nor_gate(0, 0)
logic_gate.print_output('NOR')

