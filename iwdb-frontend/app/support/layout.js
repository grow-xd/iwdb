
import React from 'react';

export const metadata = {
    title: "Support Center",
    description: "This is the page to resolve problems of the users or take their feedback and the users can connect with us from this page "
}

export default function SupportLayout({children}) {
    return (
        <div className="">
            {children}
        </div>
    )
}
