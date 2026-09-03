-- Export the immutable pre-takeover AP history from the
-- 20260817_v14_reduce100_precedents_vis Odoo snapshot.
--
-- The snapshot contains exactly 304 posted vendor bills from 2025-11-01
-- through 2026-04-30.  Each output line is one JSON document containing the
-- complete bill-facing header, product lines, attachments, and every linked
-- mail.message (comment/email/notification) with recipients and attachments.
\pset tuples_only on
\pset format unaligned

WITH selected AS (
    SELECT am.*
    FROM account_move AS am
    WHERE am.move_type = 'in_invoice'
      AND am.state = 'posted'
      AND am.invoice_date BETWEEN DATE '2025-11-01' AND DATE '2026-04-30'
)
SELECT jsonb_build_object(
    'source', 'offline_bills_messages',
    'snapshot', '20260817_v14_reduce100_precedents_vis',
    'bill_id', am.id,
    'header', jsonb_build_object(
        'id', am.id,
        'ref', am.ref,
        'vendor', jsonb_build_object('id', rp.id, 'name', rp.name),
        'invoice_origin', am.invoice_origin,
        'amount_total', am.amount_total,
        'amount_untaxed', am.amount_untaxed,
        'amount_tax', am.amount_tax,
        'payment_term', jsonb_build_object('id', apt.id, 'name', apt.name),
        'invoice_date', am.invoice_date,
        'accounting_date', am.date,
        'due_date', am.invoice_date_due,
        'payment_reference', am.payment_reference,
        'narration', am.narration,
        'journal', jsonb_build_object('id', aj.id, 'name', aj.name),
        'currency', jsonb_build_object('id', rc.id, 'name', rc.name),
        'fiscal_position', jsonb_build_object('id', afp.id, 'name', afp.name),
        'partner_bank', jsonb_build_object('id', rpb.id, 'account_number', rpb.acc_number),
        'state', am.state,
        'invoice_source_email', am.invoice_source_email,
        'delivery_date', am.delivery_date
    ),
    'lines', COALESCE((
        SELECT jsonb_agg(jsonb_build_object(
            'id', aml.id,
            'name', aml.name,
            'product', jsonb_build_object(
                'id', pp.id,
                'name', COALESCE(pt.name ->> 'en_US', pt.name::text)
            ),
            'account', jsonb_build_object(
                'id', aa.id,
                'code_by_company', aa.code_store,
                'name', COALESCE(aa.name ->> 'en_US', aa.name::text)
            ),
            'quantity', aml.quantity,
            'unit_of_measure', jsonb_build_object(
                'id', uu.id,
                'name', COALESCE(uu.name ->> 'en_US', uu.name::text)
            ),
            'price_unit', aml.price_unit,
            'discount', aml.discount,
            'price_subtotal', aml.price_subtotal,
            'taxes', COALESCE((
                SELECT jsonb_agg(jsonb_build_object('id', at.id, 'name', at.name) ORDER BY at.id)
                FROM account_move_line_account_tax_rel rel
                JOIN account_tax at ON at.id = rel.account_tax_id
                WHERE rel.account_move_line_id = aml.id
            ), '[]'::jsonb),
            'analytic_distribution', aml.analytic_distribution,
            'purchase_line_id', aml.purchase_line_id,
            'date_maturity', aml.date_maturity,
            'discount_date', aml.discount_date,
            'discount_amount_currency', aml.discount_amount_currency
        ) ORDER BY aml.id)
        FROM account_move_line aml
        LEFT JOIN product_product pp ON pp.id = aml.product_id
        LEFT JOIN product_template pt ON pt.id = pp.product_tmpl_id
        LEFT JOIN account_account aa ON aa.id = aml.account_id
        LEFT JOIN uom_uom uu ON uu.id = aml.product_uom_id
        WHERE aml.move_id = am.id AND aml.display_type = 'product'
    ), '[]'::jsonb),
    'attachments', COALESCE((
        SELECT jsonb_agg(jsonb_build_object(
            'id', ia.id,
            'name', ia.name,
            'mimetype', ia.mimetype,
            'file_size', ia.file_size
        ) ORDER BY ia.id)
        FROM ir_attachment ia
        WHERE ia.res_model = 'account.move' AND ia.res_id = am.id
    ), '[]'::jsonb),
    'messages', COALESCE((
        SELECT jsonb_agg(jsonb_build_object(
            'id', mm.id,
            'message_type', mm.message_type,
            'subject', mm.subject,
            'body_html', mm.body,
            'author', jsonb_build_object('id', author.id, 'name', author.name),
            'date', mm.date,
            'recipients', COALESCE((
                SELECT jsonb_agg(jsonb_build_object('id', recipient.id, 'name', recipient.name) ORDER BY recipient.id)
                FROM mail_message_res_partner_rel mmr
                JOIN res_partner recipient ON recipient.id = mmr.res_partner_id
                WHERE mmr.mail_message_id = mm.id
            ), '[]'::jsonb),
            'attachments', COALESCE((
                SELECT jsonb_agg(jsonb_build_object(
                    'id', ma.id,
                    'name', ma.name,
                    'mimetype', ma.mimetype,
                    'file_size', ma.file_size
                ) ORDER BY ma.id)
                FROM message_attachment_rel mar
                JOIN ir_attachment ma ON ma.id = mar.attachment_id
                WHERE mar.message_id = mm.id
            ), '[]'::jsonb)
        ) ORDER BY mm.date, mm.id)
        FROM mail_message mm
        LEFT JOIN res_partner author ON author.id = mm.author_id
        WHERE mm.model = 'account.move' AND mm.res_id = am.id
    ), '[]'::jsonb)
)::text
FROM selected am
LEFT JOIN res_partner rp ON rp.id = am.partner_id
LEFT JOIN account_payment_term apt ON apt.id = am.invoice_payment_term_id
LEFT JOIN account_journal aj ON aj.id = am.journal_id
LEFT JOIN res_currency rc ON rc.id = am.currency_id
LEFT JOIN account_fiscal_position afp ON afp.id = am.fiscal_position_id
LEFT JOIN res_partner_bank rpb ON rpb.id = am.partner_bank_id
ORDER BY am.invoice_date, am.id;
