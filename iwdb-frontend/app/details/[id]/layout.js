import React from 'react';

export const metadata = {
    title: "Details",
    description: "Here the users will find the all the details of a website or link"
}

export default function DetailsLayout({children}) {
    return (
        <div className="">
            {children}
        </div>
    )
}
